from elasticsearch import Elasticsearch
from core.config import settings

client = Elasticsearch(
    [settings.ELASTIC_URL],
    basic_auth=(settings.ELASTIC_USERNAME, settings.ELASTIC_PASSWORD),
    ca_certs="core/ca.crt"
)

# Kiểm tra kết nối khi khởi động
if client.ping():
    print("✅ Connected to Elasticsearch!")
else:
    print("❌ Failed to connect to Elasticsearch!")



