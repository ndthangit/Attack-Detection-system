from pydantic import  BaseModel
from pydantic_settings import BaseSettings

class GANParameters(BaseModel):
    input_size: int = 256
    hidden_size_g: int = 31
    hidden_size_d: int = 42
    dropout_rate_g: float = 0.42
    dropout_rate_d: float = 0.32
    learning_rate_g: float = 0.008
    learning_rate_d: float = 0.009
    num_epochs_g: int = 5
    batch_size: int = 74
    seq_len: int = 10

class RLParameters(BaseModel):
    latent_size: int = 128
    learning_rate_mlp: float = 0.005
    num_episodes: int = 5
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

