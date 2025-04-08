import torch
from transformers import DistilBertTokenizer, DistilBertModel, pipeline


class DistilBERTFeatureExtractor:
    def __init__(self, model_name="distilbert-base-uncased"):
        # Kiểm tra và sử dụng GPU nếu có
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load tokenizer và model của DistilBERT
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)

    def extract_features(self, text):
        """Trích xuất đặc trưng từ DistilBERT, đảm bảo tất cả tensor ở cùng một thiết bị."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Chuyển tất cả tensor đầu vào lên thiết bị (GPU nếu có)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)  # Chạy model trên GPU

        # Lấy embedding của token [CLS], chuyển về CPU trước khi lưu
        sentence_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().squeeze()
        return sentence_embedding



