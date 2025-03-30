import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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