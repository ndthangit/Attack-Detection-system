from pydantic import  BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Load biến môi trường từ file .env
load_dotenv()

class DatabaseSettings(BaseSettings):
    DB_URL: str = os.getenv("DB_URL")
    DB_NAME: str = os.getenv("DB_NAME")
    DB_USER: str = os.getenv("DB_USER")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD")
    DB_PORT: int = os.getenv("DB_PORT")
    DB_HOST: str = os.getenv("DB_HOST")

# graph layer
class GraphCreatorParameters(BaseModel):
    num_features: int = 768
    hidden_channels: int = 256
    feature_sim_threshold: float = 0.7
    time_threshold: int = 60
    k_nearest: int = 5
    seq_len: int = 10

class GraphTrainingParameters(BaseModel):
    epochs: int = 1
    eval_every: int = 1

class GraphConfig(BaseSettings):
    creator: GraphCreatorParameters = GraphCreatorParameters()
    training: GraphTrainingParameters = GraphTrainingParameters()


database_settings = DatabaseSettings()
# print(database_settings)


