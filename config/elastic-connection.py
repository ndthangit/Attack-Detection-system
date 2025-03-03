from operator import index

from elasticsearch import Elasticsearch

# Kết nối đến Elasticsearch
client = Elasticsearch("http://localhost:9200")

# # Kiểm tra kết nối
# if es.ping():
#     print("Connected to Elasticsearch")
# else:
#     print("Elasticsearch connection failed")


resp = client.indices.create(
    index="bgl",
)
print(resp)
resp = client.cluster.put_component_template(
    name="component_template1",
    template={
        "mappings": {
            "properties": {
                "@timestamp": {
                    "type": "date"
                }
            }
        }
    },
)
print(resp)

resp1 = client.cluster.put_component_template(
    name="runtime_component_template",
    template={
        "mappings": {
            "runtime": {
                "day_of_week": {
                    "type": "keyword",
                    "script": {
                        "source": "emit(doc['@timestamp'].value.dayOfWeekEnum.getDisplayName(TextStyle.FULL, Locale.ENGLISH))"
                    }
                }
            }
        }
    },
)
print(resp1)