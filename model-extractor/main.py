from fastapi import FastAPI
from sqlalchemy import text

from LLMs.DistilBert import DistilBERTFeatureExtractor
from core.elastic import client
from core.database import engine
import core.config as configuration
import pandas as pd

from model.PreTrainingLayer import PreTrainingLayer

app = FastAPI()

@app.get("/")
async def root():
    return { f"num_sample: hello"}

@app.get("/exacted-data/")
async def get_exacted_data(schema: str, table: str):
    query = f'SELECT * FROM "exacted-feature"."{schema}"."{table}" LIMIT 1;'
    with engine.connect() as connection:
        result = connection.execute(text(query))
        columns = result.keys()
        data = [dict(zip(columns, row)) for row in result.fetchall()]
    return { "data": data }

@app.get("/exacting")
async def get_exacting():

    response = client.cat.indices(index="*", h="index", format="json")
    data_processing = PreTrainingLayer()
    extractor = DistilBERTFeatureExtractor()

    # print("Các index người dùng (không phải hệ thống):")
    # for name in user_indices:
    #     print(name)

    user_indices = ['aminer-fox-test']

    for index_name in user_indices:
        print(f"Đang xử lý index: {index_name}")
        # Kiểm tra xem bảng có tồn tại không trước khi xóa
        with engine.connect() as conn:
            # Kiểm tra xem bảng có tồn tại không trước khi xóa
            if engine.dialect.has_table(conn, index_name):
                conn.execute(text(f'DROP TABLE IF EXISTS "{index_name}"'))
                conn.commit()
                print(f"Đã xóa bảng {index_name}")
            else:
                print(f"Bảng {index_name} không tồn tại")

        query = {
            "query": {
                "match_all": {}
            }

        }
        scroll_size = 100
        scroll_timeout = '5m'  # Tăng thời gian scroll

        try:
            response = client.search(
                index=index_name,
                body=query,
                scroll=scroll_timeout,
                size=scroll_size
            )
            scroll_id = response['_scroll_id']
            total_samples = 0

            while True:
                if not response["hits"]["hits"]:
                    break

                try:
                    data_json = [hit["_source"] for hit in response["hits"]["hits"]]
                    # print(data_json)
                    data_raw = pd.DataFrame(data_json)

                    # Áp dụng format_sample_to_text để tạo hai cột
                    text_and_timestamps = data_raw.apply(lambda row: data_processing.format_sample_to_text(row), axis=1)
                    data_raw['text_representation'] = text_and_timestamps.apply(lambda x: x[0])
                    data_raw['timestamps'] = text_and_timestamps.apply(lambda x: x[1][0])

                    # print(data_raw.columns)
                    # print(data_raw['text_representation'])

                    data = extractor.extract_features(data_raw['text_representation'].tolist())
                    data = pd.DataFrame(data)
                    data['timestamps'] = data_raw['timestamps'].tolist()
                    data['Label'] = data_raw['Label'].tolist()

                    data = pd.DataFrame(data)
                    # print(data)
                    # print(data.columns)
                    data.to_sql(
                        name=index_name,
                        con=engine,
                        index=False,  # Không ghi index của DataFrame
                        if_exists='append'  # Thay thế nếu bảng đã tồn tại
                    )

                except Exception as batch_error:
                    print(f"Error processing batch: {str(batch_error)}")
                    continue

                # Lấy batch tiếp theo
                try:
                    response = client.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
                except Exception as scroll_error:
                    print(f"Scroll error: {str(scroll_error)}")
                    break

        finally:
            if 'scroll_id' in locals() and scroll_id:
                try:
                    client.clear_scroll(scroll_id=scroll_id)
                except Exception as clear_error:
                    print(f"Error clearing scroll: {str(clear_error)}")
    return { "message": "success" }



