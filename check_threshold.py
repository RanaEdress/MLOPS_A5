import mlflow
import sys

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy", 0)
print("Accuracy:", accuracy)

if accuracy < 0.85:
    print("FAILED: Accuracy below threshold")
    sys.exit(1)
else:
    print("PASSED: Accuracy OK")