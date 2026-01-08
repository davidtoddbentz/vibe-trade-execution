"""
Container Manager for Local Strategy Execution.

Manages Docker containers for running strategies locally.
This helps understand what Docker provides and how to control it.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ContainerInfo:
    """Information about a running strategy container."""

    container_id: str
    strategy_id: str
    algorithm_name: str
    status: str
    created_at: datetime
    ports: dict[str, str]
    image: str


class ContainerManager:
    """Manages Docker containers for local strategy execution.

    What Docker Provides:
    - Isolation: Each strategy runs in its own container
    - Resource Limits: CPU/memory constraints per container
    - Networking: Isolated network for each container
    - Volumes: Persistent data storage
    - Logs: Container stdout/stderr
    - Health Checks: Monitor container status

    Controls Available:
    - Start: docker run
    - Stop: docker stop
    - Remove: docker rm
    - Logs: docker logs
    - Stats: docker stats
    - Inspect: docker inspect
    """

    def __init__(self, project_root: Path | None = None):
        """Initialize container manager.

        Args:
            project_root: Project root directory
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root
        self.lean_image = "quantconnect/lean:latest"

    def start_strategy(
        self,
        strategy_id: str,
        algorithm_name: str,
        algorithm_code: str,
        config: dict[str, Any] | None = None,
    ) -> ContainerInfo:
        """Start a strategy in a Docker container.

        Args:
            strategy_id: Unique strategy identifier
            algorithm_name: Name of LEAN algorithm class
            algorithm_code: Python code for the algorithm
            config: Optional configuration (data feeds, etc.)

        Returns:
            ContainerInfo about the running container
        """
        # Create strategy-specific directory
        strategy_dir = self.project_root / "lean" / "strategies" / strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # Write algorithm code
        algorithm_file = strategy_dir / f"{algorithm_name}.py"
        algorithm_file.write_text(algorithm_code)

        # Build container name
        container_name = f"vibe-trade-{strategy_id}"

        # Build Docker command
        cmd = [
            "docker",
            "run",
            "-d",  # Detached mode
            "--name",
            container_name,
            "-v",
            f"{strategy_dir}:/Lean/Algorithm.Python",
            "-v",
            f"{self.project_root / 'lean' / 'Data'}:/Data",
            "-v",
            f"{self.project_root / 'lean' / 'Results' / strategy_id}:/Results",
            # Add environment variables
            "-e",
            f"STRATEGY_ID={strategy_id}",
            "-e",
            f"ALGORITHM_NAME={algorithm_name}",
            self.lean_image,
            "dotnet",
            "QuantConnect.Lean.Launcher.dll",
            "--algorithm-type-name",
            algorithm_name,
            "--algorithm-language",
            "Python",
            "--algorithm-location",
            "/Lean/Algorithm.Python",
        ]

        # Add config if provided
        if config:
            for key, value in config.items():
                cmd.extend(["-e", f"{key}={value}"])

        logger.info(f"Starting container for strategy: {strategy_id}")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            container_id = result.stdout.strip()

            # Get container info
            info = self.get_container_info(container_id)
            logger.info(f"✅ Container started: {container_id[:12]}")
            return info

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start container: {e.stderr}")
            raise

    def stop_strategy(self, strategy_id: str) -> bool:
        """Stop a running strategy container.

        Args:
            strategy_id: Strategy identifier

        Returns:
            True if stopped successfully
        """
        container_name = f"vibe-trade-{strategy_id}"

        try:
            subprocess.run(
                ["docker", "stop", container_name],
                check=True,
                capture_output=True,
            )
            logger.info(f"✅ Container stopped: {container_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop container: {e.stderr}")
            return False

    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy container.

        Args:
            strategy_id: Strategy identifier

        Returns:
            True if removed successfully
        """
        container_name = f"vibe-trade-{strategy_id}"

        try:
            # Stop first if running
            subprocess.run(
                ["docker", "stop", container_name],
                capture_output=True,
            )
            # Remove
            subprocess.run(
                ["docker", "rm", container_name],
                check=True,
                capture_output=True,
            )
            logger.info(f"✅ Container removed: {container_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to remove container: {e.stderr}")
            return False

    def get_logs(self, strategy_id: str, tail: int = 100) -> str:
        """Get logs from a strategy container.

        Args:
            strategy_id: Strategy identifier
            tail: Number of lines to return

        Returns:
            Container logs
        """
        container_name = f"vibe-trade-{strategy_id}"

        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(tail), container_name],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return ""

    def get_stats(self, strategy_id: str) -> dict[str, Any]:
        """Get resource usage stats for a container.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Stats dictionary (CPU, memory, etc.)
        """
        container_name = f"vibe-trade-{strategy_id}"

        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "json", container_name],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout:
                return json.loads(result.stdout)
            return {}
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return {}

    def list_running_strategies(self) -> list[ContainerInfo]:
        """List all running strategy containers.

        Returns:
            List of ContainerInfo
        """
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=vibe-trade-", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                check=True,
            )
            container_ids = [cid.strip() for cid in result.stdout.split("\n") if cid.strip()]

            containers = []
            for container_id in container_ids:
                try:
                    info = self.get_container_info(container_id)
                    containers.append(info)
                except Exception as e:
                    logger.warning(f"Failed to get info for {container_id}: {e}")

            return containers
        except subprocess.CalledProcessError:
            return []

    def get_container_info(self, container_id: str) -> ContainerInfo:
        """Get detailed information about a container.

        Args:
            container_id: Container ID or name

        Returns:
            ContainerInfo
        """
        try:
            result = subprocess.run(
                ["docker", "inspect", container_id],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)[0]

            # Extract strategy_id from container name
            container_name = data["Name"].lstrip("/")
            strategy_id = container_name.replace("vibe-trade-", "")

            # Extract ports
            ports = {}
            if "NetworkSettings" in data and "Ports" in data["NetworkSettings"]:
                for port, mapping in data["NetworkSettings"]["Ports"].items():
                    if mapping:
                        ports[port] = mapping[0]["HostPort"]

            return ContainerInfo(
                container_id=container_id[:12],
                strategy_id=strategy_id,
                algorithm_name=data["Config"]["Env"].get("ALGORITHM_NAME", "Unknown"),
                status=data["State"]["Status"],
                created_at=datetime.fromisoformat(data["Created"].replace("Z", "+00:00")),
                ports=ports,
                image=data["Config"]["Image"],
            )
        except Exception as e:
            logger.error(f"Failed to inspect container {container_id}: {e}")
            raise
