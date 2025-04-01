import mlflow.pyfunc
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Kiểm tra và sử dụng GPU nếu có
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")

# Load tokenizer và model của DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)


def extract_features(text):
    """Trích xuất đặc trưng từ DistilBERT, đảm bảo tất cả tensor ở cùng một thiết bị."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Chuyển tất cả tensor đầu vào lên thiết bị (GPU nếu có)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)  # Chạy model trên GPU

    # Lấy embedding của token [CLS], chuyển về CPU trước khi lưu
    sentence_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().squeeze()
    return sentence_embedding

class DistilBertFeatureExtractor(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Sử dụng thiết bị: {self.device}")

    def load_context(self, context):
        """Load tokenizer và model khi MLflow khởi tạo"""
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(self.device)
        self.model.eval()  # Chế độ inference

    def predict(self, context, input_texts):
        """Xử lý batch text và trả về embeddings"""
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        inputs = self.tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Lấy embedding của token [CLS] và chuyển về CPU
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        return embeddings.squeeze()  # Giảm chiều nếu đầu vào là 1 text