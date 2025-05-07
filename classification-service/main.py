import mlflow
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
import httpx
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch import nn

from classifier.CustomPWAGAT import CustomPWAGAT
from core.config import ClassifierParameters

app = FastAPI()


async def fetch_training_data():
    """
    Fetch training data from the external endpoint, handling list structure.

    Returns:
        tuple: (data_x, data_timestamps, data_y)
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get('http://localhost:8005/training')
            response.raise_for_status()
            data = pd.DataFrame(response.json()['data'])
            data_x = data.iloc[:, :-2]
            data_y = data.iloc[:, -1]
            data_timestamps = data['timestamps']
            print(data_y.unique())

            # Encode labels
            label_encoder = LabelEncoder()
            data_y_encoded = pd.Series(label_encoder.fit_transform(data_y), name='Label')

            return data_x, data_timestamps, data_y, data_y_encoded

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"HTTP error occurred: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching data: {e}")


def split_data(data_x, data_timestamps, data_y, train_ratio=0.5):
    """
    Split data into training (first 70%) and testing (last 30%).

    Args:
        data_x (pd.DataFrame): Features
        data_timestamps (pd.DataFrame): Timestamps
        data_y (pd.Series): Labels
        train_ratio (float): Proportion of data for training

    Returns:
        tuple: (train_x, train_timestamps, train_y, test_x, test_timestamps, test_y)
    """
    total_samples = len(data_x)
    train_size = int(total_samples * train_ratio)

    # Sequential split: first 70% for training, last 30% for testing
    train_x = data_x.iloc[:train_size]
    train_timestamps = data_timestamps.iloc[:train_size]
    train_y = data_y.iloc[:train_size]

    test_x = data_x.iloc[train_size:]
    test_timestamps = data_timestamps.iloc[train_size:]
    test_y = data_y.iloc[train_size:]

    return train_x, train_timestamps, train_y, test_x, test_timestamps, test_y


def evaluate_model(model, test_x, test_timestamps, test_y, label_encoder):
    """
    Evaluate the model on the test set and return metrics.

    Args:
        model (CustomPWAGAT): Trained model
        test_x (pd.DataFrame): Test features
        test_timestamps (pd.DataFrame): Test timestamps
        test_y (pd.Series): Test labels (encoded)
        label_encoder (LabelEncoder): Encoder for decoding labels

    Returns:
        dict: Evaluation metrics (e.g., accuracy, loss)
    """
    predicted_labels = model.predict(test_x, test_timestamps)
    # Decode predictions and true labels for accuracy
    predicted_labels_decoded = label_encoder.inverse_transform(predicted_labels)
    test_y_decoded = label_encoder.inverse_transform(test_y)
    accuracy = accuracy_score(test_y_decoded, predicted_labels_decoded)

    features, labels, delta_t = model.gan_system.preprocess_data(test_x, test_timestamps, test_y)
    dataset = model.gan_system.create_dataset(features, delta_t, labels, model.seq_len)
    test_loader = model.gan_system.create_dataloader(dataset, batch_size=32, shuffle=False)

    model.gan_system.generator.eval()
    model.gan_system.discriminator.eval()

    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch_features, batch_delta_t, batch_labels in test_loader:
            batch_features = batch_features.to(model.device, non_blocking=True)
            batch_delta_t = batch_delta_t.to(model.device, non_blocking=True)
            fake_data, _ = model.gan_system.generator(batch_features, batch_delta_t)
            reconstruction_loss = nn.MSELoss()(fake_data, batch_features)
            total_loss += reconstruction_loss.item() * batch_features.size(0)
            total_samples += batch_features.size(0)

    avg_loss = total_loss / total_samples

    return {"test_accuracy": accuracy, "test_loss": avg_loss}
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/train")
async def train_model_endpoint(batch_size: int = 10000, seq_len: int = 10):
    """
    Train and evaluate the CustomPWAGAT model, logging to MLflow.

    Args:
        batch_size (int): Size of each training batch
        seq_len (int): Sequence length for the model

    Returns:
        dict: Training and evaluation results, including MLflow run ID and metrics
    """
    try:
        # Set MLflow experiment
        mlflow.set_experiment("CustomPWAGAT_Training")

        param = ClassifierParameters()

        with mlflow.start_run() as run:
            # Log parameters
            # Prepare dictionaries with prefixed keys
            gan_params = {f"GAN_{key}": value for key, value in param.gan.__dict__.items()}
            rl_params = {f"RL_{key}": value for key, value in param.rl.__dict__.items()}

            # Log all parameters at once
            mlflow.log_params(gan_params)
            mlflow.log_params(rl_params)


            # Initialize the model
            model = CustomPWAGAT(seq_len=seq_len)
            model.create_gan_model(batch_size=batch_size)
            model.create_rl_agent()

            # Fetch and split data
            data_x, data_timestamps, data_y, label_encoder = await fetch_training_data()
            # print(data_x.head())

            train_x, train_timestamps, train_y, test_x, test_timestamps, test_y = split_data(
                data_x, data_timestamps, data_y
            )
            print(train_y)

            # Log dataset sizes
            mlflow.log_metric("train_samples", len(train_x))
            mlflow.log_metric("test_samples", len(test_x))

            # Train the model
            model.fit(
                data_train_x=train_x,
                data_timestamps=train_timestamps,
                data_train_y=train_y,
                val_data=(test_x, test_timestamps, test_y)
            )

            # Log training metrics
            for epoch, (gan_loss, rl_reward) in enumerate(
                    zip(model.training_history["gan_loss"], model.training_history["rl_rewards"])
            ):
                mlflow.log_metric("gan_loss", gan_loss, step=epoch)
                mlflow.log_metric("rl_reward", rl_reward, step=epoch)

            # Evaluate on test set
            metrics = evaluate_model(model, test_x, test_timestamps, test_y, label_encoder)
            mlflow.log_metric("test_accuracy", metrics["test_accuracy"])
            mlflow.log_metric("test_loss", metrics["test_loss"])

            # Save and log the model
            model_path = "custom_pwagat_model.pth"
            model.save_model(model_path)
            mlflow.pytorch.log_model(model, "model")

            return {
                "status": "success",
                "mlflow_run_id": run.info.run_id,
                "metrics": metrics,
                "model_path": model_path
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
