from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Load biến môi trường từ file .env
load_dotenv()

class Settings(BaseSettings):
    ELASTIC_URL: str = os.getenv("ELASTIC_URL")
    ELASTIC_USERNAME: str = os.getenv("ELASTIC_USERNAME")
    ELASTIC_PASSWORD: str = os.getenv("ELASTIC_PASSWORD")

class DatabaseSettings(BaseSettings):
    DB_URL: str = os.getenv("DB_URL")
    DB_NAME: str = os.getenv("DB_NAME")
    DB_USER: str = os.getenv("DB_USER")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD")
    DB_PORT: int = os.getenv("DB_PORT")
    DB_HOST: str = os.getenv("DB_HOST")

class MlflowSettings(BaseSettings):
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME")
    MLFLOW_RUN: str = os.getenv("MLFLOW_RUN")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_S3_ENDPOINT_URL: str = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")


database_settings = DatabaseSettings()
# print(database_settings)

settings = Settings()
# mlflow_settings = MlflowSettings()
# print(settings)