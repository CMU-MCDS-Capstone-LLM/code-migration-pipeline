import os
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

    def should_run(self) -> bool:
        """Run if no trajectory exists or needs updating"""
        trajectory_file = (
            self.config.trajectory_path / f"{self.config.identifier}.patch"
        )
        return not trajectory_file.exists()

    def load_cached_result(self) -> str:
        """Load existing trajectory path"""
        return str(self.config.trajectory_path / f"{self.config.identifier}.patch")

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

        # Set environment variables for SWE-agent
        env_vars = {
            "REPO_NAME": self.config.commit_info.folder_name,
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "WORKSPACE_DIR": "/ws",  # Container path
            "PROBLEM_STATEMENT": self.get_problem_statement(),
        }

        # Build docker run command with environment variables
        # Use the same named volume as the repo container for a shared workspace
        shared_volume = "repos-ws"

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

        return str(self.config.trajectory_path / f"{self.config.identifier}.patch")
