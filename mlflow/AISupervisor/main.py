from dotenv import load_dotenv
import os
from mlflow.tracking import MlflowClient
from fastapi import FastAPI
import mlflow.pyfunc

load_dotenv()
app = FastAPI()

client = MlflowClient()
experiment_name = "AI_Supervisor_Experiment"

# Kết nối đến MLflow Server
MODEL_NAME = "ai-supervisor_model"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
print(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_registry_uri("s3://bucket")


@app.post("/")
async def supervise(model_name: str, accuracy: float, loss: float):
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)
        # Log artifact nếu có
    return {"message": f"Supervised {model_name} successfully"}


