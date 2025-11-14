import hashlib
import os
import subprocess
import tempfile
import time
import logging
from pathlib import Path

from ..pipeline.config import MigConfig
from ..pipeline.task import Task, TaskId
from ..models import Repo
from ..tasks.repo_dl import RepoDlTask

logger = logging.getLogger(__name__)


class DockerContainerTask(Task[str]):
    """Manages Docker container for a specific repository"""

    def __init__(self, config: MigConfig, repo_dl: RepoDlTask) -> None:
        task_id = TaskId(f"docker_container_{config.identifier}")
        super().__init__(task_id, config, depends_on=[repo_dl])
        self.repo_dl = repo_dl
        self.container_name = f"repo-{config.commit_info.folder_name}"

        folder_hash = hashlib.md5(config.commit_info.folder_name.encode()).hexdigest()[
            :8
        ]
        self.network_alias = f"repo-{config.commit_info.folder_name[:30]}-{folder_hash}"
        if len(self.network_alias) > 63:
            self.network_alias = f"repo-{folder_hash}"
        self.dockerfile_path = config.env_path / "Dockerfile"

    def should_run(self) -> bool:
        """Run if container doesn't exist or needs rebuilding"""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    f"name={self.container_name}",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return self.container_name not in result.stdout
        except subprocess.CalledProcessError:
            return True

    def load_cached_result(self) -> str:
        """Return existing container name"""
        return self.container_name

    def _ensure_lsp_repograph_server(
        self, container_name: str, repo_path_in_container: str, network_alias: str
    ):
        """Ensure LSP-RepoGraph server is running in the container."""
        server_port = 8000
        server_url = f"http://{network_alias}:{server_port}"

        # Ensure repo directory is writable (needed for scratch files)
        # Permissions are already set during copy, but ensure they're correct
        # Make world-writable so any container user can read/write
        make_writable_cmd = [
            "docker",
            "exec",
            "-u",
            "root",  # Run as root to fix permissions
            container_name,
            "bash",
            "-c",
            f"chmod -R a+rwX {repo_path_in_container} 2>/dev/null || true",
        ]
        subprocess.run(make_writable_cmd, check=False, timeout=30)

        # Clean up old scratch files from repo directory (always cleanup, even if server is running)
        cleanup_cmd = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-c",
            f"find {repo_path_in_container} -name '_scratch_*.py' -delete 2>/dev/null || true",
        ]
        subprocess.run(cleanup_cmd, check=False, timeout=30)

        # Quick check if port 8000 is listening (server is running)
        port_check = subprocess.run(
            [
                "docker",
                "exec",
                container_name,
                "bash",
                "-c",
                f"python3 -c \"import socket; s = socket.socket(); s.settimeout(1); result = s.connect_ex(('localhost', {server_port})); s.close(); exit(0 if result == 0 else 1)\" 2>/dev/null",
            ],
            capture_output=True,
            timeout=3,
        )
        if port_check.returncode == 0:
            logger.info(f"LSP-RepoGraph server already running at {server_url}")
            return

        logger.info(f"Starting LSP-RepoGraph server in container {container_name}")

        # Verify mount exists (simple check)
        mount_check = subprocess.run(
            ["docker", "exec", container_name, "test", "-d", "/lsp-repograph"],
            capture_output=True,
            timeout=5,
        )
        if mount_check.returncode != 0:
            logger.error("LSP-Repograph not mounted at /lsp-repograph")
            return

        # Check filesystem space first
        logger.info("Checking filesystem space...")
        df_cmd = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-c",
            "df -h / | tail -1",
        ]
        df_result = subprocess.run(
            df_cmd, check=False, capture_output=True, text=True, timeout=10
        )
        if df_result.returncode == 0:
            logger.info(f"Filesystem usage: {df_result.stdout.strip()}")
            # Check if filesystem is nearly full (>= 95%)
            if "9[0-9]%" in df_result.stdout or "100%" in df_result.stdout:
                logger.warning(
                    "Filesystem is nearly full! You may need to clean up Docker images/volumes."
                )
                logger.warning("Run: docker system prune -a --volumes -f")

        # Install dependencies (use Python 3.11 for LSP-Repograph)
        # Use --break-system-packages for Python 3.11 (safe in containers, required for Debian 12)
        logger.info("Installing lsp-repograph dependencies (Python 3.11)...")
        install_cmd = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-c",
            "cd /lsp-repograph && "
            "python3.11 -m pip install --no-cache-dir --break-system-packages -r requirements.txt 2>&1",
        ]
        install_result = subprocess.run(
            install_cmd, check=False, capture_output=True, text=True, timeout=300
        )
        if install_result.returncode == 0:
            logger.info("✓ Installed LSP-Repograph dependencies")
        else:
            logger.error(
                f"Failed to install LSP-Repograph dependencies (exit code: {install_result.returncode})"
            )
            logger.error(f"Installation output: {install_result.stdout[-1000:]}")
            logger.error(f"Installation error: {install_result.stderr[-1000:]}")
            # Try fallback: install without version pins
            logger.info(
                "Trying fallback installation (without version pins, Python 3.11)..."
            )
            fallback_cmd = [
                "docker",
                "exec",
                container_name,
                "bash",
                "-c",
                "cd /lsp-repograph && "
                "grep -v '^#' requirements.txt | grep -v '^$' | "
                "sed 's/==.*$//; s/>=.*$//; s/<=.*$//; s/~=.*$//; s/!=.*$//' | "
                "xargs python3.11 -m pip install --no-cache-dir --break-system-packages 2>&1",
            ]
            fallback_result = subprocess.run(
                fallback_cmd, check=False, capture_output=True, text=True, timeout=300
            )
            if fallback_result.returncode == 0:
                logger.info("✓ Installed LSP-Repograph dependencies (fallback method)")
            else:
                logger.error(
                    f"Fallback installation also failed (exit code: {fallback_result.returncode})"
                )
                logger.error(f"Fallback output: {fallback_result.stdout[-1000:]}")
                logger.error(f"Fallback error: {fallback_result.stderr[-1000:]}")
                raise RuntimeError(
                    "Failed to install LSP-Repograph dependencies. Cannot start server."
                )

        # Install server dependencies (Flask, Waitress) - Python 3.11
        logger.info("Installing server dependencies (Flask, Waitress) - Python 3.11...")
        server_deps_cmd = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-c",
            "python3.11 -m pip install --no-cache-dir --break-system-packages flask waitress 2>&1",
        ]
        server_deps_result = subprocess.run(
            server_deps_cmd, check=False, capture_output=True, text=True, timeout=60
        )
        if server_deps_result.returncode == 0:
            logger.info("✓ Installed server dependencies")
        else:
            logger.error(
                f"Failed to install server dependencies (exit code: {server_deps_result.returncode})"
            )
            logger.error(f"Output: {server_deps_result.stdout[-500:]}")
            logger.error(f"Error: {server_deps_result.stderr[-500:]}")
            raise RuntimeError(
                "Failed to install server dependencies (Flask, Waitress). Cannot start server."
            )

        # Verify critical dependencies are installed (Python 3.11)
        logger.info("Verifying dependencies are installed (Python 3.11)...")
        verify_cmd = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-c",
            "python3.11 -c 'import multilspy; import flask; import waitress; print(\"All dependencies available\")' 2>&1",
        ]
        verify_result = subprocess.run(
            verify_cmd, check=False, capture_output=True, text=True, timeout=10
        )
        if verify_result.returncode == 0:
            logger.info("✓ Verified all dependencies are available")
        else:
            logger.error(
                f"Dependency verification failed (exit code: {verify_result.returncode})"
            )
            logger.error(f"Verification output: {verify_result.stdout}")
            logger.error(f"Verification error: {verify_result.stderr}")
            raise RuntimeError(
                "Critical dependencies (multilspy, flask, waitress) are not available. Cannot start server."
            )

        # Start server in background (using Python 3.11)
        # Use infra server (wraps LSP-Repograph without modifying it)
        # Set TMPDIR to repo path so scratch files go there (writable location)
        # Jedi LSP server will use Python 3.11 - it's a static analysis tool that works on any Python repo
        logger.info("Starting server (Python 3.11)...")
        start_cmd = [
            "docker",
            "exec",
            "-d",
            container_name,
            "bash",
            "-c",
            f"export PATH=$HOME/.local/bin:$PATH && "
            f"export PYTHONPATH=/lsp-repograph:$PYTHONPATH && "
            f"export TMPDIR={repo_path_in_container} && "
            f"python3.11 /infra/utils/repograph_server.py "
            f"--repo-path '{repo_path_in_container}' "
            f"--port {server_port} "
            f"--host 0.0.0.0 "
            f"> /tmp/lsp-repograph-server.log 2>&1",
        ]
        subprocess.run(start_cmd, check=False, timeout=10)
        time.sleep(3)  # Give server time to start

        # Verify server is listening on port 8000
        for i in range(5):
            port_check = subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "bash",
                    "-c",
                    f"python3 -c \"import socket; s = socket.socket(); s.settimeout(1); result = s.connect_ex(('localhost', {server_port})); s.close(); exit(0 if result == 0 else 1)\" 2>/dev/null",
                ],
                capture_output=True,
                timeout=3,
            )
            if port_check.returncode == 0:
                logger.info(
                    f"✓ LSP-RepoGraph server started and listening on port {server_port} "
                    f"(accessible at {server_url} from containers on swe-network)"
                )
                return
            time.sleep(1)

        # Port not listening, check logs
        logger.warning("Server port not listening, checking logs...")
        log_cmd = [
            "docker",
            "exec",
            container_name,
            "cat",
            "/tmp/lsp-repograph-server.log",
        ]
        log_result = subprocess.run(log_cmd, capture_output=True, text=True, timeout=5)
        if log_result.stdout:
            logger.error(f"Server logs:\n{log_result.stdout}")
        else:
            logger.error("Server failed to start (no logs available)")

    def run(self) -> str:
        """Ensure tests image exists, then ensure repo container is running."""

        # Create network if it doesn't exist (idempotent)
        subprocess.run(
            ["docker", "network", "create", "swe-network"], check=False
        )  # Don't fail if network exists

        # Image tag for tests stage; build only if it doesn't exist yet
        base_image_name = f"{self.config.commit_info.repo_safe}:tests"
        img_id = subprocess.run(
            ["docker", "images", "-q", base_image_name],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

        if not img_id:
            logger.info(f"Building Docker image: {base_image_name}")
            subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    base_image_name,
                    "-f",
                    str(self.dockerfile_path),
                    str(self.config.repo_path),
                ],
                check=True,
            )
        # No volume setup here; keep container self-contained (image has code)

        # Ensure a shared named volume exists for ALL repos
        # This allows switching repos without reseeding volumes
        volume_name = "repos-ws"
        subprocess.run(["docker", "volume", "create", volume_name], check=False)

        # Always copy from clean source to ensure we start with pristine repo state
        # This ensures migrations don't persist across runs
        # Fix permissions immediately after copying so all containers can write
        logger.info(
            f"Copying clean repo from {self.config.repo_path.parent} to volume..."
        )
        repo_name = self.config.commit_info.folder_name
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{volume_name}:/ws",
                "-v",
                f"{str(self.config.repo_path.parent)}:/src:ro",
                "alpine",
                "sh",
                "-c",
                f"rm -rf /ws/{repo_name} && "
                f"cp -a /src/{repo_name} /ws/ && "
                f"chmod -R a+rwX /ws/{repo_name}",
            ],
            check=False,
        )

        # Ensure container exists and is running
        try:
            existing = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    f"name={self.container_name}",
                    "--format",
                    "{{.Status}}",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            existing = ""

        # Get LSP-Repograph path for mounting
        lsp_repograph_path = Path(__file__).parent.parent.parent / "LSP-Repograph"
        lsp_repograph_path = lsp_repograph_path.resolve()

        # Get infra path for mounting (contains repograph_server.py)
        infra_path = Path(__file__).parent.parent
        infra_path = infra_path.resolve()

        # The repo is mounted at /ws, and the repo name is the folder name
        repo_name = self.config.commit_info.folder_name
        repo_path_in_container = f"/ws/{repo_name}"

        if not existing:
            docker_run_cmd = [
                "docker",
                "run",
                "-d",
                "--name",
                self.container_name,
                "--network",
                "swe-network",
                "--network-alias",
                self.network_alias,  # Shorter alias for DNS resolution
                "-v",
                f"{volume_name}:/ws",
                "-v",
                f"{lsp_repograph_path}:/lsp-repograph:ro",  # Mount as read-only
                "-v",
                f"{infra_path}:/infra:ro",  # Mount infra folder (contains server)
                base_image_name,
                "sleep",
                "infinity",
            ]
            subprocess.run(docker_run_cmd, check=True)

            # Wait a moment for container to be ready
            time.sleep(1)

            # Permissions are already set during copy (world-writable), but ensure they're correct
            logger.info("Ensuring workspace permissions are correct...")
            fix_perms_cmd = [
                "docker",
                "exec",
                "-u",
                "root",  # Run as root to fix permissions
                self.container_name,
                "bash",
                "-c",
                f"chmod -R a+rwX {repo_path_in_container} 2>/dev/null || true",
            ]
            subprocess.run(fix_perms_cmd, check=False, timeout=30)

            # Skip repository dependency installation - they're already in the Docker image
            # The image has dependencies installed during build, so no need to install again
            logger.info(
                "Skipping repository dependency installation (already in Docker image)"
            )
        elif not existing.lower().startswith("up"):
            subprocess.run(["docker", "start", self.container_name], check=True)
            time.sleep(1)

            # Permissions are already set during copy (world-writable), but ensure they're correct
            logger.info("Ensuring workspace permissions are correct...")
            fix_perms_cmd = [
                "docker",
                "exec",
                "-u",
                "root",  # Run as root to fix permissions
                self.container_name,
                "bash",
                "-c",
                f"chmod -R a+rwX {repo_path_in_container} 2>/dev/null || true",
            ]
            subprocess.run(fix_perms_cmd, check=False, timeout=30)

            # If container was created before, we need to add the network alias
            # Check if alias exists, if not add it
            try:
                # Get current network aliases
                inspect_cmd = [
                    "docker",
                    "inspect",
                    "--format",
                    "{{range .NetworkSettings.Networks}}{{range .Aliases}}{{.}} {{end}}{{end}}",
                    self.container_name,
                ]
                result = subprocess.run(
                    inspect_cmd, capture_output=True, text=True, check=True
                )
                existing_aliases = result.stdout.strip().split()

                if self.network_alias not in existing_aliases:
                    # Connect to network with alias
                    network_connect_cmd = [
                        "docker",
                        "network",
                        "connect",
                        "--alias",
                        self.network_alias,
                        "swe-network",
                        self.container_name,
                    ]
                    subprocess.run(
                        network_connect_cmd, check=False
                    )  # May fail if already connected, that's OK
            except Exception as e:
                logger.warning(f"Could not ensure network alias: {e}")

            # Skip repository dependency installation - they're already in the Docker image
            # The image has dependencies installed during build, so no need to install again
            logger.info(
                "Skipping repository dependency installation (already in Docker image)"
            )

        # Start LSP-RepoGraph server in the container
        self._ensure_lsp_repograph_server(
            self.container_name, repo_path_in_container, self.network_alias
        )

        return self.container_name
