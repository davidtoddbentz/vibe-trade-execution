#!/usr/bin/env python3
"""Run a test backtest on Cloud Run."""

import json
import time
from datetime import datetime

from google.cloud import run_v2, storage

# Configuration
PROJECT_ID = "vibe-trade-475704"
REGION = "us-central1"
LEAN_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/vibe-trade-lean/vibe-trade-lean:latest"
RESULTS_BUCKET = "vibe-trade-backtest-results"
DATA_BUCKET = "batch-save"


def create_test_strategy_ir() -> dict:
    """Create a simple EMA crossover strategy IR for testing."""
    return {
        "strategy_id": "test-cloud-001",
        "strategy_name": "Cloud Test: EMA Crossover",
        "symbol": "BTC-USD",
        "resolution": "Hour",
        "indicators": [
            {"id": "ema_fast", "type": "EMA", "period": 9},
            {"id": "ema_slow", "type": "EMA", "period": 21},
        ],
        "state": [],
        "entry": {
            "id": "entry_ema_cross",
            "condition": {
                "type": "compare",
                "left": {"type": "indicator", "indicator_id": "ema_fast"},
                "op": ">",
                "right": {"type": "indicator", "indicator_id": "ema_slow"},
            },
            "action": {"type": "set_holdings", "allocation": 0.95},
            "on_fill": [],
        },
        "exits": [
            {
                "id": "exit_ema_cross",
                "priority": 1,
                "condition": {
                    "type": "compare",
                    "left": {"type": "indicator", "indicator_id": "ema_fast"},
                    "op": "<",
                    "right": {"type": "indicator", "indicator_id": "ema_slow"},
                },
                "action": {"type": "liquidate"},
            }
        ],
        "gates": [],
        "on_bar": [],
        "on_bar_invested": [],
    }


def upload_strategy_ir(backtest_id: str, ir: dict) -> str:
    """Upload strategy IR to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(RESULTS_BUCKET)

    ir_path = f"backtests/{backtest_id}/strategy_ir.json"
    blob = bucket.blob(ir_path)
    blob.upload_from_string(json.dumps(ir, indent=2), content_type="application/json")

    print(f"✅ Uploaded strategy IR to gs://{RESULTS_BUCKET}/{ir_path}")
    return f"gs://{RESULTS_BUCKET}/{ir_path}"


def create_and_run_job(backtest_id: str, ir_gcs_path: str) -> str:
    """Create and run Cloud Run Job."""
    jobs_client = run_v2.JobsClient()

    job_name = f"backtest-{backtest_id[:8]}"
    parent = f"projects/{PROJECT_ID}/locations/{REGION}"

    # Environment variables
    env_vars = [
        run_v2.EnvVar(name="BACKTEST_ID", value=backtest_id),
        run_v2.EnvVar(name="STRATEGY_IR_GCS_PATH", value=ir_gcs_path),
        run_v2.EnvVar(name="RESULTS_GCS_PATH", value=f"gs://{RESULTS_BUCKET}/backtests/{backtest_id}/results.json"),
        run_v2.EnvVar(name="DATA_BUCKET", value=DATA_BUCKET),
        run_v2.EnvVar(name="SYMBOL", value="BTC-USD"),
        run_v2.EnvVar(name="START_DATE", value="20260101"),
        run_v2.EnvVar(name="END_DATE", value="20260108"),
        run_v2.EnvVar(name="INITIAL_CASH", value="100000"),
    ]

    # Container
    container = run_v2.Container(
        image=LEAN_IMAGE,
        env=env_vars,
        resources=run_v2.ResourceRequirements(
            limits={"cpu": "2", "memory": "4Gi"},
        ),
    )

    # Task template
    task_template = run_v2.TaskTemplate(
        containers=[container],
        max_retries=0,
        timeout="1800s",  # 30 min timeout
    )

    # Execution template
    execution_template = run_v2.ExecutionTemplate(
        template=task_template,
        task_count=1,
        parallelism=1,
    )

    # Job
    job = run_v2.Job(
        template=execution_template,
        labels={
            "app": "vibe-trade",
            "component": "backtest",
            "backtest-id": backtest_id[:63],
        },
    )

    print(f"🚀 Creating Cloud Run Job: {job_name}")

    # Create job
    try:
        operation = jobs_client.create_job(parent=parent, job=job, job_id=job_name)
        created_job = operation.result()
        print(f"✅ Job created: {created_job.name}")
    except Exception as e:
        if "already exists" in str(e):
            print("⚠️  Job already exists, deleting and recreating...")
            jobs_client.delete_job(name=f"{parent}/jobs/{job_name}").result()
            time.sleep(2)
            operation = jobs_client.create_job(parent=parent, job=job, job_id=job_name)
            created_job = operation.result()
            print(f"✅ Job recreated: {created_job.name}")
        else:
            raise

    # Run job
    print("▶️  Running job...")
    run_operation = jobs_client.run_job(name=created_job.name)
    execution = run_operation.result()
    print(f"✅ Execution started: {execution.name}")

    return execution.name


def wait_for_completion(execution_name: str, timeout: int = 600) -> dict:
    """Wait for job execution to complete."""
    executions_client = run_v2.ExecutionsClient()
    start_time = time.time()

    print(f"⏳ Waiting for job completion (timeout: {timeout}s)...")

    while time.time() - start_time < timeout:
        execution = executions_client.get_execution(name=execution_name)

        # Check conditions
        for condition in execution.conditions:
            if condition.type_ == "Completed":
                if condition.state == run_v2.Condition.State.CONDITION_SUCCEEDED:
                    print("✅ Job completed successfully!")
                    return {"status": "completed", "execution": execution}
                elif condition.state == run_v2.Condition.State.CONDITION_FAILED:
                    print(f"❌ Job failed: {condition.message}")
                    return {"status": "failed", "execution": execution, "error": condition.message}

        # Check if running
        if execution.running_count > 0:
            elapsed = int(time.time() - start_time)
            print(f"   Running... ({elapsed}s elapsed)", end="\r")

        time.sleep(5)

    print(f"⚠️  Timeout after {timeout}s")
    return {"status": "timeout"}


def fetch_results(backtest_id: str) -> dict | None:
    """Fetch results from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(RESULTS_BUCKET)

    # Try strategy_output.json first (our custom output)
    blob = bucket.blob(f"backtests/{backtest_id}/strategy_output.json")
    if blob.exists():
        content = blob.download_as_text()
        return json.loads(content)

    # Try results.json
    blob = bucket.blob(f"backtests/{backtest_id}/results.json")
    if blob.exists():
        content = blob.download_as_text()
        return json.loads(content)

    return None


def fetch_logs(backtest_id: str) -> str | None:
    """Fetch logs from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(RESULTS_BUCKET)

    blob = bucket.blob(f"backtests/{backtest_id}/log.txt")
    if blob.exists():
        return blob.download_as_text()

    return None


def main():
    backtest_id = f"test-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"🧪 Running Cloud Run Backtest: {backtest_id}")
    print(f"{'='*60}\n")

    # Step 1: Create and upload strategy IR
    ir = create_test_strategy_ir()
    ir_gcs_path = upload_strategy_ir(backtest_id, ir)

    # Step 2: Create and run job
    execution_name = create_and_run_job(backtest_id, ir_gcs_path)

    # Step 3: Wait for completion
    result = wait_for_completion(execution_name)

    print(f"\n{'='*60}")
    print("📊 RESULTS")
    print(f"{'='*60}\n")

    if result["status"] == "completed":
        # Fetch and display results
        results = fetch_results(backtest_id)
        if results:
            print(json.dumps(results, indent=2))
        else:
            print("⚠️  No results file found")

        # Fetch and display logs
        logs = fetch_logs(backtest_id)
        if logs:
            print(f"\n{'='*60}")
            print("📝 LOGS (last 50 lines)")
            print(f"{'='*60}\n")
            print("\n".join(logs.split("\n")[-50:]))
    else:
        print(f"Status: {result['status']}")
        if "error" in result:
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
