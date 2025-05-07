import numpy as np
import pandas as pd
from fastapi import FastAPI
import logging
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import inspect

from WebAttackGraph.AttackBehaviorGNN import model
from core.config import GraphConfig
from core.database import engine

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/training")
async def graph_embedding_training():
    try:
        # Lấy danh sách tất cả các bảng trong cơ sở dữ liệu
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        table_names = ['aminer-fox-test']
        logger.info(f"Danh sách bảng trong cơ sở dữ liệu: {table_names}")

        graphParameter = GraphConfig()
        result_data = {}  # Dictionary để lưu kết quả embedding theo tên bảng

        for table_name in table_names:
            logger.info(f"Bắt đầu xử lý bảng: {table_name}")
            label_encoder = LabelEncoder()

            try:
                # Tải dữ liệu từ bảng
                query = f'SELECT * FROM "{table_name}";'
                data_raw = pd.read_sql(query, engine)
                logger.info(f"Dữ liệu từ bảng {table_name} đã được tải: {data_raw.shape}")

                if data_raw.empty:
                    logger.warning(f"Bảng {table_name} không có dữ liệu")
                    continue

                # Kiểm tra các cột cần thiết
                required_cols = ['timestamps', 'Label']
                missing_cols = [col for col in required_cols if col not in data_raw.columns]
                if missing_cols:
                    logger.error(f"Bảng {table_name} thiếu các cột: {missing_cols}")
                    continue

                # Sắp xếp theo thời gian và gán nhãn
                data_raw = data_raw.sort_values(by='timestamps')
                data_raw.loc[data_raw['Label'] != 'benign', 'Label'] = 'malicious'
                original_labels = data_raw['Label'].copy()  # Lưu nhãn gốc
                data_raw['Label'] = label_encoder.fit_transform(data_raw['Label'])

                if len(data_raw) < model.seq_len:
                    logger.warning(f"Dữ liệu quá ít để xử lý bảng {table_name}")
                    continue

                # Huấn luyện và embedding toàn bộ dữ liệu
                batch_size = 100
                num_batches = (len(data_raw) + batch_size - 1) // batch_size

                # Lưu trữ embedding và thông tin bổ sung
                all_embeddings = []
                all_timestamps = []
                all_labels = []

                for batch_num in range(num_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min((batch_num + 1) * batch_size, len(data_raw))
                    batch = data_raw.iloc[start_idx:end_idx]

                    if len(batch) < model.seq_len:
                        logger.warning(f"Batch {batch_num + 1} quá nhỏ để huấn luyện")
                        continue

                    logger.info(f"Huấn luyện batch {batch_num + 1}/{num_batches} với {len(batch)} mẫu")

                    model.prepare_data(
                        batch,
                        feature_cols=None,
                        timestamp_col="timestamps",
                        label_col="Label"
                    )
                    model.train(
                        epochs=graphParameter.training.epochs,
                        eval_every=graphParameter.training.eval_every
                    )

                    # Lấy embedding từ layer cuối
                    embeddings = model.get_embeddings(layer="last")
                    all_embeddings.append(embeddings)

                    # Lưu timestamps và nhãn gốc tương ứng với batch
                    all_timestamps.append(batch['timestamps'].values)
                    all_labels.append(original_labels.iloc[start_idx:end_idx].values)

                # Kết hợp tất cả embedding, timestamps và labels
                if all_embeddings:
                    all_embeddings = np.concatenate(all_embeddings, axis=0)
                    all_timestamps = np.concatenate(all_timestamps, axis=0)
                    all_labels = np.concatenate(all_labels, axis=0)

                    # Tạo DataFrame chứa embedding, timestamps và labels
                    embedding_df = pd.DataFrame(all_embeddings,
                                               columns=[f"emb_{i}" for i in range(all_embeddings.shape[1])])
                    embedding_df['timestamps'] = all_timestamps
                    embedding_df['Label'] = all_labels

                    # Lưu kết quả embedding vào result_data với key là tên bảng
                    result_data[table_name] = embedding_df.to_dict(orient='records')
                else:
                    result_data[table_name] = []

            except Exception as e:
                logger.error(f"Lỗi khi xử lý bảng {table_name}: {str(e)}")
                result_data[table_name] = []
                continue

    except Exception as e:
        logger.error(f"Lỗi tổng quát trong quá trình huấn luyện và kiểm tra: {str(e)}")
        return {"message": f"Error during training and testing: {str(e)}"}

    return {
        "message": "training completed successfully",
        "data": result_data
    }