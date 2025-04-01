from fastapi import FastAPI
from core.elastic import client
from core.database import engine
import export_sentence.export_save
import export_sentence
import mlflow
import core.config as configuration
from export_sentence.distil_bert import DistilBertFeatureExtractor

mlflow.set_tracking_uri(configuration.MlflowSettings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(configuration.mlflow_settings.MLFLOW_EXPERIMENT_NAME)

app = FastAPI()

@app.get("/")
async def root():
    num_sample=export_sentence.export_save.export_feature_save("aminer-fox")

    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="distilbert_embedder",
            python_model=DistilBertFeatureExtractor(),
            registered_model_name="distilbert-embedder",
            extra_pip_requirements=["torch", "transformers"]  # Đảm bảo dependencies
        )

        # Đánh dấu model là production-ready ngay lập tức
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="distilbert-embedder",
            version=1,
            stage="Production"
        )

    return { f"num_sample: {num_sample}"}
