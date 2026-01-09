"""Cloud Run Jobs service for running backtests."""

import json
import logging
from datetime import datetime
from typing import Any

from google.cloud import run_v2, storage

from src.translator.ir_translator import IRTranslator, TranslationError

logger = logging.getLogger(__name__)


class CloudRunJobService:
    """Service for managing Cloud Run Jobs for backtests."""

    def __init__(
        self,
        project_id: str,
        region: str = "us-central1",
        lean_image: str | None = None,
        results_bucket: str = "vibe-trade-backtest-results",
        data_bucket: str = "batch-save",
        job_service_account: str | None = None,
    ):
        """Initialize Cloud Run Job service.

        Args:
            project_id: GCP project ID
            region: Cloud Run region
            lean_image: Full image URL for LEAN container (e.g., us-docker.pkg.dev/project/repo/image:tag)
            results_bucket: GCS bucket for backtest results
            data_bucket: GCS bucket with market data
            job_service_account: Service account email for Cloud Run Jobs
        """
        self.project_id = project_id
        self.region = region
        self.lean_image = (
            lean_image
            or f"{region}-docker.pkg.dev/{project_id}/vibe-trade-lean/vibe-trade-lean:latest"
        )
        self.results_bucket = results_bucket
        self.data_bucket = data_bucket
        self.job_service_account = (
            job_service_account
            or f"vibe-trade-lean-job-runner@{project_id}.iam.gserviceaccount.com"
        )

        self.jobs_client = run_v2.JobsClient()
        self.storage_client = storage.Client()

    async def submit_backtest_job(
        self,
        backtest_id: str,
        strategy: Any,
        cards: dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        symbol: str = "BTC-USD",
        initial_cash: float = 100000.0,
    ) -> dict[str, Any]:
        """Submit a backtest as a Cloud Run Job.

        This method:
        1. Translates strategy to IR
        2. Uploads IR JSON to GCS
        3. Creates and executes a Cloud Run Job
        4. Returns job info for status tracking

        Args:
            backtest_id: Unique ID for this backtest
            strategy: Strategy model
            cards: Dict mapping card_id to Card objects
            start_date: Backtest start date
            end_date: Backtest end date
            symbol: Trading symbol
            initial_cash: Initial capital

        Returns:
            Dict with job status and info
        """
        try:
            # Step 1: Translate strategy to IR
            logger.info(f"Backtest {backtest_id}: Translating strategy to IR")
            try:
                translator = IRTranslator(strategy, cards)
                ir = translator.translate()
            except TranslationError as e:
                return {
                    "status": "error",
                    "error": f"Translation failed: {e}",
                }

            # Step 2: Upload IR JSON to GCS
            ir_json = ir.model_dump_json(indent=2)
            ir_gcs_path = f"backtests/{backtest_id}/strategy_ir.json"

            logger.info(
                f"Backtest {backtest_id}: Uploading IR to gs://{self.results_bucket}/{ir_gcs_path}"
            )
            bucket = self.storage_client.bucket(self.results_bucket)
            blob = bucket.blob(ir_gcs_path)
            blob.upload_from_string(ir_json, content_type="application/json")

            # Step 3: Create and execute Cloud Run Job
            job_name = f"backtest-{backtest_id[:8]}"  # Keep name short
            logger.info(f"Backtest {backtest_id}: Creating Cloud Run Job {job_name}")

            job = self._create_job_spec(
                job_name=job_name,
                backtest_id=backtest_id,
                ir_gcs_path=ir_gcs_path,
                start_date=start_date,
                end_date=end_date,
                symbol=symbol,
                initial_cash=initial_cash,
            )

            # Create the job
            parent = f"projects/{self.project_id}/locations/{self.region}"
            operation = self.jobs_client.create_job(
                parent=parent,
                job=job,
                job_id=job_name,
            )

            # Wait for job creation to complete
            created_job = operation.result()
            logger.info(f"Backtest {backtest_id}: Job created: {created_job.name}")

            # Execute the job (run_job returns an Operation, we need to get the Execution from it)
            run_operation = self.jobs_client.run_job(name=created_job.name)
            execution = run_operation.result()  # Wait for execution to start
            logger.info(f"Backtest {backtest_id}: Job execution started: {execution.name}")

            return {
                "status": "submitted",
                "job_name": created_job.name,
                "execution_name": execution.name,
                "backtest_id": backtest_id,
                "ir_path": f"gs://{self.results_bucket}/{ir_gcs_path}",
                "results_path": f"gs://{self.results_bucket}/backtests/{backtest_id}/results.json",
            }

        except Exception as e:
            logger.error(f"Backtest {backtest_id}: Failed to submit job - {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }

    def _create_job_spec(
        self,
        job_name: str,
        backtest_id: str,
        ir_gcs_path: str,
        start_date: datetime,
        end_date: datetime,
        symbol: str,
        initial_cash: float,
    ) -> run_v2.Job:
        """Create Cloud Run Job specification.

        The job runs our LEAN container with:
        - Strategy IR downloaded from GCS
        - Market data fetched from GCS
        - Results uploaded to GCS on completion
        """
        # Environment variables for the LEAN container
        env_vars = [
            run_v2.EnvVar(name="BACKTEST_ID", value=backtest_id),
            run_v2.EnvVar(
                name="STRATEGY_IR_GCS_PATH", value=f"gs://{self.results_bucket}/{ir_gcs_path}"
            ),
            run_v2.EnvVar(
                name="RESULTS_GCS_PATH",
                value=f"gs://{self.results_bucket}/backtests/{backtest_id}/results.json",
            ),
            run_v2.EnvVar(name="DATA_BUCKET", value=self.data_bucket),
            run_v2.EnvVar(name="SYMBOL", value=symbol),
            run_v2.EnvVar(name="START_DATE", value=start_date.strftime("%Y%m%d")),
            run_v2.EnvVar(name="END_DATE", value=end_date.strftime("%Y%m%d")),
            run_v2.EnvVar(name="INITIAL_CASH", value=str(initial_cash)),
        ]

        # Container specification
        container = run_v2.Container(
            image=self.lean_image,
            env=env_vars,
            resources=run_v2.ResourceRequirements(
                limits={
                    "cpu": "2",
                    "memory": "4Gi",
                },
            ),
        )

        # Task template with service account
        task_template = run_v2.TaskTemplate(
            containers=[container],
            max_retries=0,  # Don't retry failed backtests
            timeout="3600s",  # 1 hour timeout
            service_account=self.job_service_account,
        )

        # Execution template
        execution_template = run_v2.ExecutionTemplate(
            template=task_template,
            task_count=1,
            parallelism=1,
        )

        # Job specification
        job = run_v2.Job(
            template=execution_template,
            labels={
                "app": "vibe-trade",
                "component": "backtest",
                "backtest-id": backtest_id[:63],  # Labels have max length
            },
        )

        return job

    async def get_job_status(self, job_name: str) -> dict[str, Any]:
        """Get the status of a Cloud Run Job.

        Args:
            job_name: Full job name (projects/*/locations/*/jobs/*)

        Returns:
            Dict with job status and results if complete
        """
        try:
            job = self.jobs_client.get_job(name=job_name)

            # Get the latest execution
            executions = list(self.jobs_client.list_executions(parent=job_name))
            if not executions:
                return {
                    "status": "unknown",
                    "error": "No executions found",
                }

            latest = executions[-1]
            execution_status = self._map_execution_status(latest)

            result = {
                "status": execution_status,
                "job_name": job_name,
                "execution_name": latest.name,
            }

            # If completed, try to fetch results from GCS
            if execution_status == "completed":
                # Extract backtest_id from job labels or name
                backtest_id = job.labels.get("backtest-id", "")
                if backtest_id:
                    results = await self._fetch_results(backtest_id)
                    if results:
                        result["results"] = results

            return result

        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    def _map_execution_status(self, execution: run_v2.Execution) -> str:
        """Map Cloud Run execution status to our BacktestStatus."""
        # Check conditions for completion/failure
        for condition in execution.conditions:
            if condition.type_ == "Completed":
                if condition.state == run_v2.Condition.State.CONDITION_SUCCEEDED:
                    return "completed"
                elif condition.state == run_v2.Condition.State.CONDITION_FAILED:
                    return "failed"

        # If not completed, check if running
        if execution.running_count > 0:
            return "running"

        return "pending"

    async def _fetch_results(self, backtest_id: str) -> dict[str, Any] | None:
        """Fetch backtest results from GCS."""
        try:
            bucket = self.storage_client.bucket(self.results_bucket)
            blob = bucket.blob(f"backtests/{backtest_id}/results.json")

            if blob.exists():
                content = blob.download_as_text()
                return json.loads(content)

            return None
        except Exception as e:
            logger.error(f"Failed to fetch results: {e}")
            return None

    async def cleanup_job(self, job_name: str) -> bool:
        """Delete a Cloud Run Job after completion.

        Args:
            job_name: Full job name

        Returns:
            True if deleted successfully
        """
        try:
            operation = self.jobs_client.delete_job(name=job_name)
            operation.result()  # Wait for deletion
            logger.info(f"Deleted job: {job_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete job {job_name}: {e}")
            return False
