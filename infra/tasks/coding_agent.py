import hashlib
import os
import shutil
import subprocess
from typing import Collection

from ..pipeline.config import MigConfig
from ..pipeline.task import Task, TaskId
from dotenv import load_dotenv


class CodingAgentTask(Task):
    """Run SWE-agent in Docker container with volume mounting for file synchronization"""

    def __init__(self, config: MigConfig, depends_on: Collection[TaskId]):
        task_id = TaskId(f"coding_agent_{config.identifier}")
        super().__init__(task_id, config, depends_on)
        self.container_name = f"repo-{config.commit_info.folder_name}"
        self.swe_agent_container_name = f"swe-agent"
        # Create the same network alias as DockerContainerTask (for repograph server URL)
        folder_hash = hashlib.md5(config.commit_info.folder_name.encode()).hexdigest()[
            :8
        ]
        self.network_alias = f"repo-{config.commit_info.folder_name[:30]}-{folder_hash}"
        # Ensure alias is <= 63 chars (Docker DNS limit)
        if len(self.network_alias) > 63:
            self.network_alias = f"repo-{folder_hash}"

    def should_run(self) -> bool:
        """Run if no trajectory exists or needs updating"""
        trajectory_file = (
            self.config.trajectory_path / f"{self.config.commit_info.folder_name}.patch"
        )
        return not trajectory_file.exists()

    def load_cached_result(self) -> str:
        """Load existing trajectory path"""
        return str(
            self.config.trajectory_path / f"{self.config.commit_info.folder_name}.patch"
        )

    def get_problem_statement(self) -> str:
        """Get problem statement from config or generate default one"""
        if self.config.problem_statement:
            return f"{self.config.problem_statement} Container name: repo-{self.config.commit_info.folder_name} (use this exact name with exec_command tool)"

        # Fallback to generating a default problem statement
        repo_name = self.config.commit_info.repo
        migration_mappings = {
            "alice-biometrics/petisco": "Migrate from FastAPI to Flask framework. Update all API endpoints, middleware, and configuration to use Flask instead of FastAPI.",
            # Add more mappings as needed
        }

        specific_migration = migration_mappings.get(
            repo_name,
            "Perform code migration as specified in the repository requirements.",
        )

        return f"Code migration task for {self.config.commit_info.folder_name}. {specific_migration} Container name: repo-{self.config.commit_info.folder_name} (use this exact name with exec_command tool)"

    def run(self) -> str:
        """Run SWE-agent in Docker container with volume mounting"""

        # Build SWE-agent image if it doesn't exist
        image_name = "swe-agent-image"
        try:
            subprocess.run(
                ["docker", "build", "-t", image_name, "./coding-agent"], check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build SWE-agent image: {e}")

        load_dotenv()

        # Ensure trajectory path exists
        self.config.trajectory_path.mkdir(parents=True, exist_ok=True)

        # Set environment variables for SWE-agent
        env_vars = {
            "REPO_NAME": self.config.commit_info.folder_name,
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "WORKSPACE_DIR": "/ws",
            "PROBLEM_STATEMENT": self.get_problem_statement(),
            "REPO_CONTAINER_NAME": self.container_name,
            "REPOGRAPH_SERVER_HOST": self.network_alias,
            "OUTPUT_DIR": "/trajectories",  # Mounted trajectory path in container
        }

        # Build docker run command with environment variables
        # Use the same named volume as the repo container for a shared workspace
        shared_volume = "repos-ws"

        # Mount trajectory path so trajectories are saved to host and overwrite old ones
        trajectory_path_in_container = "/trajectories"

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            self.swe_agent_container_name,
            "--network",
            "swe-network",  # Same network as repo container
            "-v",
            "/var/run/docker.sock:/var/run/docker.sock",  # Allow docker exec from inside agent
            "-v",
            f"{shared_volume}:/ws",  # Shared repos workspace
            "-v",
            f"{os.path.abspath('./coding-agent')}:/app",  # Mount coding-agent code (includes .git)
            "-v",
            f"{os.path.abspath('.git')}:/git",  # Mount parent .git directory for submodule references
            "-v",
            f"{self.config.trajectory_path.resolve()}:{trajectory_path_in_container}",  # Mount trajectory path
            "-w",
            "/app",  # Working directory
        ]

        # Add environment variables
        for key, value in env_vars.items():
            docker_cmd.extend(["-e", f"{key}={value}"])

        # Add the command to run
        docker_cmd.extend([image_name, "bash", "/app/start_worker.sh"])

        # Run SWE-agent in container
        try:
            subprocess.run(docker_cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SWE-agent failed: {e}")

        # Find the patch file in the trajectory directory (SWE-agent creates subdirectories)
        # SWE-agent saves to: output_dir / problem_id / problem_id.patch
        # We need to find the most recent patch file and copy it to the expected location
        patch_files = list(self.config.trajectory_path.rglob("*.patch"))
        # Use folder_name instead of identifier to avoid subdirectories from / and @ characters
        expected_patch_path = (
            self.config.trajectory_path / f"{self.config.commit_info.folder_name}.patch"
        )

        if patch_files:
            # Get the most recently modified patch file
            latest_patch = max(patch_files, key=lambda p: p.stat().st_mtime)

            # Ensure parent directory exists before copying
            expected_patch_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy it to the expected location (overwrite if exists)
            if latest_patch != expected_patch_path:
                shutil.copy2(latest_patch, expected_patch_path)
        elif not expected_patch_path.exists():
            # If no patch file found, check if SWE-agent actually ran
            # If trajectory directory is empty or has no patch files, the run might have failed
            if not any(self.config.trajectory_path.iterdir()):
                raise RuntimeError(
                    f"SWE-agent run appears to have failed. "
                    f"No trajectory files found in {self.config.trajectory_path}. "
                    f"Check the logs above for errors."
                )
            else:
                raise RuntimeError(
                    f"No patch file found in {self.config.trajectory_path}. "
                    f"Expected patch file at: {expected_patch_path}. "
                    f"Found files: {list(self.config.trajectory_path.rglob('*'))}"
                )

        return str(expected_patch_path)
