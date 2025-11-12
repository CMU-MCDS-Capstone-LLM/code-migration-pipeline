import asyncio
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from flask import Flask, request, jsonify

# Add LSP-Repograph to path (it's mounted at /lsp-repograph in container)
sys.path.insert(0, "/lsp-repograph")

from lsp_repograph.core.multilspy_client import MultilspyLSPClient

logger = logging.getLogger(__name__)

app = Flask(__name__)


def resolve_module_name(
    package_name: str,
    env_python: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> str:
    """
    Resolve a package name to its actual import module name by querying package metadata.
    Uses multiple dynamic methods without hardcoding.
    """
    import subprocess
    import shutil

    python_candidates = []
    if env_python and Path(env_python).exists():
        python_candidates.append(env_python)
    python3_path = shutil.which("python3")
    if python3_path and python3_path not in python_candidates:
        python_candidates.append(python3_path)
    python311_path = shutil.which("python3.11")
    if python311_path and python311_path not in python_candidates:
        python_candidates.append(python311_path)

    # Method 1: importlib.metadata
    script_importlib = f"""
import json
import sys
try:
    from importlib.metadata import distribution
    dist = distribution('{package_name}')
    try:
        top_level_content = dist.read_text('top_level.txt')
        if top_level_content:
            lines = top_level_content.strip().split('\\n')
            modules = [line.strip() for line in lines 
                      if line.strip() and not line.strip().startswith('#')]
            if modules:
                print(json.dumps({{'modules': modules}}))
                sys.exit(0)
    except Exception:
        pass
    if hasattr(dist, 'files') and dist.files:
        modules = set()
        for file in dist.files:
            if file.suffix == '.py' and len(file.parts) > 0:
                top_level = file.parts[0]
                if top_level.endswith('.py'):
                    top_level = top_level[:-3]
                if top_level and not top_level.startswith('_') and top_level != 'tests':
                    modules.add(top_level)
        if modules:
            preferred = [m for m in modules if not m.startswith('test')]
            print(json.dumps({{'modules': list(preferred) if preferred else list(modules)}}))
            sys.exit(0)
except Exception:
    pass
"""

    # Method 2: pkg_resources
    script_pkg_resources = f"""
import json
try:
    import pkg_resources
    dist = pkg_resources.get_distribution('{package_name}')
    if dist.has_metadata('top_level.txt'):
        modules = [line.strip() for line in dist.get_metadata('top_level.txt').split('\\n') 
                  if line.strip() and not line.strip().startswith('#')]
        if modules:
            print(json.dumps({{'modules': modules}}))
except Exception:
    pass
"""

    # Method 3: Direct import test
    package_name_underscore = package_name.replace("-", "_")
    script_import_test = f"""
import json
import sys
import importlib.util

variations = ['{package_name_underscore}', '{package_name}']
for var in variations:
    try:
        mod = __import__(var)
        actual_name = mod.__name__.split('.')[0]
        print(json.dumps({{'modules': [actual_name]}}))
        sys.exit(0)
    except ImportError:
        try:
            spec = importlib.util.find_spec(var)
            if spec is not None and spec.name:
                print(json.dumps({{'modules': [spec.name.split('.')[0]]}}))
                sys.exit(0)
        except Exception:
            continue
    except Exception:
        continue
"""

    # Try all methods
    for method_name, script in [
        ("importlib.metadata", script_importlib),
        ("pkg_resources", script_pkg_resources),
        ("direct import", script_import_test),
    ]:
        for python_interpreter in python_candidates:
            try:
                result = subprocess.run(
                    [python_interpreter, "-c", script],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        data = json.loads(result.stdout.strip())
                        modules = data.get("modules", [])
                        if modules:
                            resolved = modules[0]
                            if resolved != package_name:
                                logger.info(
                                    f"Resolved '{package_name}' -> '{resolved}'"
                                )
                            return resolved
                    except (json.JSONDecodeError, KeyError):
                        pass
            except Exception:
                continue

    logger.warning(
        f"Could not resolve '{package_name}' to import module name, using original"
    )
    return package_name


def find_file_importing_module(
    repo_root: Path, module: str, qualpath: Optional[str]
) -> Optional[Tuple[Path, int, int]]:
    """Find a file in the repo that imports the given module."""
    if qualpath:
        qualpath_escaped = re.escape(qualpath)
        patterns = [
            rf"^\s*from\s+{re.escape(module)}\s+import.*\b{qualpath_escaped}\b",
            rf"^\s*import\s+{re.escape(module)}\b",
        ]
    else:
        patterns = [
            rf"^\s*import\s+{re.escape(module)}\b",
            rf"^\s*from\s+{re.escape(module)}\s+import",
        ]

    for py_file in repo_root.rglob("*.py"):
        try:
            if any(
                skip in str(py_file)
                for skip in [".git", "__pycache__", ".pytest_cache", "node_modules"]
            ):
                continue

            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for line_num, line in enumerate(content.split("\n")):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                for pattern in patterns:
                    if re.search(pattern, stripped, re.IGNORECASE):
                        if qualpath:
                            qualpath_match = re.search(
                                rf"\b{re.escape(qualpath)}\b", stripped, re.IGNORECASE
                            )
                            if qualpath_match:
                                return (py_file, line_num, qualpath_match.start())
                        else:
                            module_match = re.search(
                                rf"\b{re.escape(module)}\b", stripped, re.IGNORECASE
                            )
                            if module_match:
                                return (py_file, line_num, module_match.start())
        except Exception:
            continue

    return None


def format_references(
    repo_root: Path, refs: Iterable[Dict[str, object]]
) -> Dict[str, object]:
    """Convert raw reference dictionaries into normalized JSON-friendly format."""
    seen_paths = set()
    files: List[str] = []
    formatted_refs: List[Dict[str, object]] = []

    for ref in refs:
        abs_path = Path(ref["absolute_path"]).resolve()
        try:
            rel_path = abs_path.relative_to(repo_root)
        except ValueError:
            continue

        if str(rel_path) not in seen_paths:
            seen_paths.add(str(rel_path))
            files.append(str(rel_path))

        formatted_refs.append(
            {
                "relative_path": str(rel_path),
                "absolute_path": str(abs_path),
                "line": ref["line"] + 1,
                "character": ref["character"] + 1,
            }
        )

    return {"files": files, "references": formatted_refs}


def build_custom_init(extra_paths: Iterable[str]) -> Optional[Dict[str, object]]:
    """Prepare initializationOptions for Multilspy/Jedi."""
    import shutil

    workspace_cfg: Dict[str, object] = {}
    normalized_extra_paths = [str(Path(p).resolve()) for p in extra_paths if p]

    python311_path = shutil.which("python3.11")
    if python311_path:
        workspace_cfg["environmentPath"] = python311_path
    else:
        python3_path = shutil.which("python3")
        if python3_path:
            workspace_cfg["environmentPath"] = python3_path

    if normalized_extra_paths:
        workspace_cfg["extraPaths"] = normalized_extra_paths

    if workspace_cfg:
        return {"initializationOptions": {"workspace": workspace_cfg}}
    return None


class RepoGraphServer:
    """HTTP server wrapper for LSP-RepoGraph functionality."""

    def __init__(
        self,
        repo_path: Path,
        env_python: Optional[str] = None,
        extra_paths: Optional[list] = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.env_python = env_python
        self.extra_paths = extra_paths or []
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="repograph"
        )

    def _run_query_in_thread(self, request_data: Dict) -> Dict:
        """Run query in a separate thread with its own event loop."""

        def _run_with_event_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return self._execute_query(request_data)
            finally:
                try:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                finally:
                    loop.close()

        future: Future = self.executor.submit(_run_with_event_loop)
        return future.result(timeout=600)

    def _execute_query(self, request_data: Dict) -> Dict:
        """Execute a RepoGraph query."""
        repo_path = Path(request_data.get("repo_path", str(self.repo_path)))
        source_module = request_data.get("source_module")
        source_qualpath = request_data.get("source_qualpath")
        env_python = request_data.get("env_python") or self.env_python
        extra_paths = request_data.get("extra_paths", self.extra_paths)
        workspace_symbols = request_data.get("workspace_symbols", [])

        custom_init = build_custom_init(extra_paths)
        client = MultilspyLSPClient(str(repo_path), custom_init_params=custom_init)

        library_consumers = {"files": [], "references": []}

        try:
            if source_module:
                resolved_module = resolve_module_name(
                    source_module, env_python=env_python, repo_root=repo_path
                )
                if resolved_module != source_module:
                    logger.info(f"Resolved '{source_module}' -> '{resolved_module}'")
                source_module = resolved_module

                import_file = find_file_importing_module(
                    repo_path, source_module, source_qualpath
                )

                if import_file:
                    file_path, line, char = import_file
                    library_refs = client.find_refs_by_loc(
                        path=str(file_path.relative_to(repo_path)),
                        line=line,
                        character=char,
                    )
                else:
                    library_refs = client.find_refs_by_fqn(
                        module=source_module, qualpath=source_qualpath
                    )

                library_consumers = format_references(repo_path, library_refs)
                logger.info(
                    f"Found {len(library_consumers.get('references', []))} references"
                )

            workspace_callers: Dict[str, Dict[str, object]] = {}
            for sym in workspace_symbols:
                if ":" in sym:
                    module_part, qualpath_part = sym.split(":", 1)
                    qualpath_part = qualpath_part or None
                else:
                    parts = sym.split(".")
                    module_part = parts[0] if parts else sym
                    qualpath_part = ".".join(parts[1:]) if len(parts) > 1 else None

                module_part = resolve_module_name(
                    module_part, env_python=env_python, repo_root=repo_path
                )

                refs = client.find_refs_by_fqn(
                    module=module_part, qualpath=qualpath_part
                )
                key = f"{module_part}:{qualpath_part or ''}"
                workspace_callers[key] = format_references(repo_path, refs)

        finally:
            client.shutdown()

        return {
            "source": {"module": source_module, "qualpath": source_qualpath},
            "library_consumers": library_consumers,
            "workspace_callers": workspace_callers,
        }


# Global server instance
_server: Optional[RepoGraphServer] = None


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {"status": "healthy", "repo_path": str(_server.repo_path) if _server else None}
    )


@app.route("/run", methods=["POST"])
def run():
    """Execute a RepoGraph query."""
    if _server is None:
        return jsonify({"error": "Server not initialized"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        result = _server._run_query_in_thread(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /run endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def create_app(
    repo_path: Path,
    env_python: Optional[str] = None,
    extra_paths: Optional[list] = None,
):
    """Create and configure Flask app."""
    global _server
    _server = RepoGraphServer(repo_path, env_python, extra_paths)
    return app


def main():
    """Main entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="LSP-RepoGraph HTTP server")
    parser.add_argument(
        "--repo-path", type=str, required=True, help="Path to the repository root"
    )
    parser.add_argument(
        "--env-python",
        type=str,
        help="Path to Python interpreter for the repo environment",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        help="Additional site-packages directories",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )

    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Starting LSP-RepoGraph server for repo: {repo_path}")

    app = create_app(
        repo_path=repo_path, env_python=args.env_python, extra_paths=args.extra_path
    )

    try:
        from waitress import serve

        serve(app, host=args.host, port=args.port, threads=4, channel_timeout=120)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
