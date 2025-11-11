import hashlib
import os
import subprocess
import tempfile
import time
import logging
from typing import Collection
from pathlib import Path

from ..pipeline.config import MigConfig
from ..pipeline.task import Task, TaskId
from ..models import Repo

logger = logging.getLogger(__name__)


class DockerContainerTask(Task):
    """Manages Docker container for a specific repository"""

    def __init__(self, config: MigConfig, depends_on: Collection[TaskId]):
        task_id = TaskId(f"docker_container_{config.identifier}")
        super().__init__(task_id, config, depends_on)
        self.container_name = f"repo-{config.commit_info.folder_name}"
        # Create a shorter network alias for DNS (Docker DNS labels limited to 63 chars)
        # Use first 30 chars of folder name + hash suffix to ensure uniqueness
        folder_hash = hashlib.md5(config.commit_info.folder_name.encode()).hexdigest()[
            :8
        ]
        self.network_alias = f"repo-{config.commit_info.folder_name[:30]}-{folder_hash}"
        # Ensure alias is <= 63 chars (Docker DNS limit)
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

        # Install dependencies
        logger.info("Installing lsp-repograph dependencies...")
        install_cmd = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-c",
            "cd /lsp-repograph && "
            "pip install -q -r requirements.txt 2>&1 || "
            "(grep -v '^#' requirements.txt | grep -v '^$' | "
            "sed 's/==.*$//; s/>=.*$//; s/<=.*$//; s/~=.*$//; s/!=.*$//' | "
            "xargs pip install -q 2>&1)",
        ]
        install_result = subprocess.run(
            install_cmd, check=False, capture_output=True, text=True, timeout=180
        )
        if install_result.returncode == 0:
            logger.info("✓ Installed dependencies")
        else:
            logger.warning("Dependency installation had issues (continuing anyway)")

        # Start server in background
        logger.info("Starting server...")
        start_cmd = [
            "docker",
            "exec",
            "-d",
            container_name,
            "bash",
            "-c",
            f"export PATH=$HOME/.local/bin:$PATH && "
            f"export PYTHONPATH=/lsp-repograph:$PYTHONPATH && "
            f"cd /lsp-repograph && "
            f"nohup python -m lsp_repograph.server "
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
            subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    base_image_name,
                    "-f",
                    str(self.dockerfile_path),
                    "--target",
                    "tests",
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
        logger.info(
            f"Copying clean repo from {self.config.repo_path.parent} to volume..."
        )
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
                f"rm -rf /ws/{self.config.commit_info.folder_name} && cp -a /src/{self.config.commit_info.folder_name} /ws/",
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
                base_image_name,
                "sleep",
                "infinity",
            ]
            subprocess.run(docker_run_cmd, check=True)

            # Wait a moment for container to be ready
            time.sleep(1)

            logger.info("Installing repository dependencies...")
            install_repo_reqs_cmd = [
                "docker",
                "exec",
                self.container_name,
                "bash",
                "-c",
                f"cd {repo_path_in_container} && "
                "if [ -f requirements/requirements.txt ]; then "
                "pip install -q -r requirements/requirements.txt 2>&1 || "
                "(echo 'Exact versions failed, trying flexible versions...' && "
                "grep -v '^#' requirements/requirements.txt | grep -v '^$' | "
                "sed 's/==.*$//; s/>=.*$//; s/<=.*$//; s/~=.*$//; s/!=.*$//' | "
                "xargs pip install -q 2>&1); "
                "elif [ -f requirements.txt ]; then "
                "pip install -q -r requirements.txt 2>&1 || "
                "(echo 'Exact versions failed, trying flexible versions...' && "
                "grep -v '^#' requirements.txt | grep -v '^$' | "
                "sed 's/==.*$//; s/>=.*$//; s/<=.*$//; s/~=.*$//; s/!=.*$//' | "
                "xargs pip install -q 2>&1); "
                "else "
                "echo 'No requirements.txt found, skipping'; "
                "fi",
            ]
            try:
                repo_reqs_result = subprocess.run(
                    install_repo_reqs_cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if repo_reqs_result.returncode == 0:
                    logger.info("✓ Installed repository dependencies")
                else:
                    logger.warning(
                        f"Repository dependency installation had issues (code: {repo_reqs_result.returncode})"
                    )
                    logger.debug(f"Output: {repo_reqs_result.stdout[-500:]}")
                    logger.debug(f"Error: {repo_reqs_result.stderr[-500:]}")
            except subprocess.TimeoutExpired:
                logger.warning("Repository dependency installation timed out")
            except Exception as e:
                logger.warning(f"Exception installing repository dependencies: {e}")
        elif not existing.lower().startswith("up"):
            subprocess.run(["docker", "start", self.container_name], check=True)
            time.sleep(1)

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

            logger.info("Installing repository dependencies...")
            install_repo_reqs_cmd = [
                "docker",
                "exec",
                self.container_name,
                "bash",
                "-c",
                f"cd {repo_path_in_container} && "
                "if [ -f requirements/requirements.txt ]; then "
                "pip install -q -r requirements/requirements.txt 2>&1 || "
                "(echo 'Exact versions failed, trying flexible versions...' && "
                "grep -v '^#' requirements/requirements.txt | grep -v '^$' | "
                "sed 's/==.*$//; s/>=.*$//; s/<=.*$//; s/~=.*$//; s/!=.*$//' | "
                "xargs pip install -q 2>&1); "
                "elif [ -f requirements.txt ]; then "
                "pip install -q -r requirements.txt 2>&1 || "
                "(echo 'Exact versions failed, trying flexible versions...' && "
                "grep -v '^#' requirements.txt | grep -v '^$' | "
                "sed 's/==.*$//; s/>=.*$//; s/<=.*$//; s/~=.*$//; s/!=.*$//' | "
                "xargs pip install -q 2>&1); "
                "else "
                "echo 'No requirements.txt found, skipping'; "
                "fi",
            ]
            try:
                repo_reqs_result = subprocess.run(
                    install_repo_reqs_cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if repo_reqs_result.returncode == 0:
                    logger.info("✓ Installed/updated repository dependencies")
                else:
                    logger.warning(
                        f"Repository dependency installation had issues (code: {repo_reqs_result.returncode})"
                    )
                    logger.debug(f"Output: {repo_reqs_result.stdout[-500:]}")
                    logger.debug(f"Error: {repo_reqs_result.stderr[-500:]}")
            except subprocess.TimeoutExpired:
                logger.warning("Repository dependency installation timed out")
            except Exception as e:
                logger.warning(f"Exception installing repository dependencies: {e}")

        # Start LSP-RepoGraph server in the container
        self._ensure_lsp_repograph_server(
            self.container_name, repo_path_in_container, self.network_alias
        )

        return self.container_name
