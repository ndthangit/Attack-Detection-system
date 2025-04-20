from elasticsearch import Elasticsearch
from core.config import settings

client = Elasticsearch(
    [settings.ELASTIC_URL],  # Địa chỉ Elasticsearch
    basic_auth=(settings.ELASTIC_USERNAME, settings.ELASTIC_PASSWORD),  # Thay your_password bằng ELASTIC_PASSWORD
    # verify_certs=False  # Bỏ kiểm tra SSL (hoặc cung cấp đường dẫn CA nếu cần)
    ca_certs="core/ca.crt",
    request_timeout=60

)

# Kiểm tra kết nối khi khởi động
if client.ping():
    print("✅ Connected to Elasticsearch!")
else:
    print("❌ Failed to connect to Elasticsearch!")