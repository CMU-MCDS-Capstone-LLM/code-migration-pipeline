import os
import subprocess
import tempfile
from typing import Collection
from pathlib import Path

from ..pipeline.config import MigConfig
from ..pipeline.task import Task, TaskId
from ..models import Repo


class DockerContainerTask(Task):
    """Manages Docker container for a specific repository"""

    def __init__(self, config: MigConfig, depends_on: Collection[TaskId]):
        task_id = TaskId(f"docker_container_{config.identifier}")
        super().__init__(task_id, config, depends_on)
        self.container_name = f"repo-{config.commit_info.folder_name}"
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

        # Seed the volume with the entire data/repos folder on first use (includes .git)
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
                "-lc",
                'if [ -z "$(ls -A /ws 2>/dev/null)" ]; then cp -a /src/. /ws/; fi',
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

        if not existing:
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    self.container_name,
                    "--network",
                    "swe-network",
                    "-v",
                    f"{volume_name}:/ws",
                    base_image_name,
                    "sleep",
                    "infinity",
                ],
                check=True,
            )
        elif not existing.lower().startswith("up"):
            subprocess.run(["docker", "start", self.container_name], check=True)

        return self.container_name
