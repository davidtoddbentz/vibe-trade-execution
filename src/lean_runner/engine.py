"""
LEAN Engine Wrapper.

Provides a Python interface to run LEAN algorithms via Docker.
This is for local development and testing.
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LeanEngine:
    """Wrapper for running LEAN algorithms via Docker."""
    
    def __init__(self, lean_image: str = "quantconnect/lean:latest"):
        """Initialize LEAN engine.
        
        Args:
            lean_image: Docker image name for LEAN
        """
        self.lean_image = lean_image
        self.project_root = Path(__file__).parent.parent.parent
    
    def run_backtest(
        self,
        algorithm_name: str,
        start_date: str = "20240101",
        end_date: str = "20241231",
        cash: float = 100000.0,
        data_directory: Optional[str] = None,
        results_directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a backtest using LEAN.
        
        Args:
            algorithm_name: Name of algorithm class (e.g., "SimpleMaCrossover")
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            cash: Starting cash
            data_directory: Path to data directory (relative to project root)
            results_directory: Path to results directory (relative to project root)
            
        Returns:
            Results dictionary with status and output paths
        """
        # Set up paths
        algorithms_dir = self.project_root / "lean" / "Algorithms"
        if data_directory:
            data_dir = self.project_root / data_directory
        else:
            data_dir = self.project_root / "lean" / "Data"
        if results_directory:
            results_dir = self.project_root / results_directory
        else:
            results_dir = self.project_root / "lean" / "Results"
        
        # Create directories if they don't exist
        results_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Build Docker command
        # LEAN expects specific volume mounts and parameters
        # Note: Dates and capital are set in algorithm code, not via CLI
        # Note: LEAN image entrypoint is already the launcher, so we just pass arguments
        # algorithm-location should point to the directory containing the Python file
        # LEAN will look for a file matching the algorithm-type-name
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{algorithms_dir}:/Lean/Algorithm.Python",
            "-v", f"{data_dir}:/Data",
            "-v", f"{results_dir}:/Results",
            self.lean_image,
            "--algorithm-type-name", algorithm_name,
            "--algorithm-language", "Python",
            "--algorithm-location", f"/Lean/Algorithm.Python/{algorithm_name}.py",
            "--data-folder", "/Data",
            "--results-destination-folder", "/Results",
        ]
        
        logger.info(f"Running LEAN backtest: {algorithm_name}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,  # Don't raise on non-zero exit
            )
            
            if result.returncode != 0:
                logger.error(f"LEAN backtest failed with exit code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return {
                    "status": "error",
                    "exit_code": result.returncode,
                    "stderr": result.stderr,
                    "stdout": result.stdout,
                }
            
            # Try to find and parse results
            results_file = results_dir / "backtest-results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results_data = json.load(f)
                    return {
                        "status": "success",
                        "results": results_data,
                        "results_path": str(results_file),
                    }
            
            return {
                "status": "success",
                "stdout": result.stdout,
                "results_path": str(results_dir),
            }
            
        except Exception as e:
            logger.error(f"Error running LEAN: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }
    
    def check_docker_available(self) -> bool:
        """Check if Docker is available and LEAN image exists.
        
        Returns:
            True if Docker and LEAN image are available
        """
        try:
            # Check Docker
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"Docker available: {result.stdout.strip()}")
            
            # Check if LEAN image exists
            result = subprocess.run(
                ["docker", "images", "-q", self.lean_image],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout.strip():
                logger.info(f"LEAN image found: {self.lean_image}")
                return True
            else:
                logger.warning(f"LEAN image not found: {self.lean_image}")
                logger.info(f"Run: docker pull {self.lean_image}")
                return False
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Docker not available: {e}")
            return False

