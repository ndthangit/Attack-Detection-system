from pydantic import  BaseModel
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

# graph layer
class GraphCreatorParameters(BaseModel):
    num_features: int = 768
    hidden_channels: int = 256
    feature_sim_threshold: float = 0.7
    time_threshold: int = 60
    k_nearest: int = 5

class GraphTrainingParameters(BaseModel):
    epochs: int = 100
    eval_every: int = 10

class GraphConfig(BaseSettings):
    creator: GraphCreatorParameters = GraphCreatorParameters()
    training: GraphTrainingParameters = GraphTrainingParameters()

class GANParameters(BaseModel):
    input_size: int = 768
    hidden_size_g: int = 31
    hidden_size_d: int = 42
    dropout_rate_g: float = 0.42
    dropout_rate_d: float = 0.32
    learning_rate_g: float = 0.008
    learning_rate_d: float = 0.009
    num_epochs_g: int = 25
    batch_size: int = 74
    seq_len: int = 10

class RLParameters(BaseModel):
    latent_size: int = 128
    learning_rate_mlp: float = 0.005
    num_episodes: int = 25
    batch_size: int = 74


class ClassifierParameters(BaseModel):
    gan: GANParameters = GANParameters()
    rl: RLParameters = RLParameters()


class Parameters(BaseSettings):
    input_size: int = 768
    hidden_size_g: int = 31
    hidden_size_d: int = 42
    latent_size: int = 128
    output_size: int = 768
    dropout_rate_g: float = 0.42
    dropout_rate_d: float = 0.32
    learning_rate_g: float = 0.008
    learning_rate_d: float = 0.009
    num_epochs_g: int = 25  # 152

    hidden_size_mlp: int = 53
    dropout_rate_mlp: float = 0.31
    learning_rate_mlp: float = 0.005
    num_episodes: int = 25  # 245
    batch_size_mlp: int = 74  # Đồng bộ với giá trị đã xác định trước đó
    batch_size_rl: int = 74  # Đồng bộ batch_size_rl với batch_size_mlp


database_settings = DatabaseSettings()
# print(database_settings)

settings = Settings()

# print(settings)

