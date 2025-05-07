from operator import index
from fastapi import APIRouter
from sqlalchemy import inspect
from core.config import GraphConfig, ClassifierParameters
from core.database import engine
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from purposed_model.CustomPWAGAT import CustomPWAGAT
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/", tags=["Experiments"])
async def training_and_testing():
    """
    Huấn luyện và kiểm tra mô hình CustomPWAGAT trên tất cả các bảng trong cơ sở dữ liệu.
    Sử dụng TimeSeriesSplit để chia dữ liệu thành tập huấn luyện và kiểm tra.
    """
    try:
        # Khởi tạo mô hình
        model = CustomPWAGAT(seq_len=10)
        logger.info("Khởi tạo mô hình CustomPWAGAT thành công")

        # Lấy danh sách tất cả các bảng trong cơ sở dữ liệu
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        logger.info(f"Danh sách bảng trong cơ sở dữ liệu: {table_names}")

        # Khởi tạo cấu hình
        graphParameter = GraphConfig()
        classifierParameter = ClassifierParameters()

        # Dictionary để lưu kết quả kiểm tra
        results = {}

        for table_name in table_names:
            logger.info(f"Bắt đầu xử lý bảng: {table_name}")
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

                # Sắp xếp dữ liệu theo thời gian
                data_raw = data_raw.sort_values(by='timestamps')
                data_raw.loc[data_raw['Label'] != 'Benign', 'Label'] = 'Malicious'

                # Chia dữ liệu thành tập huấn luyện và kiểm tra sử dụng TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=5)
                for fold, (train_idx, test_idx) in enumerate(tscv.split(data_raw)):
                    logger.info(f"Xử lý fold {fold + 1} cho bảng {table_name}")
                    train_data = data_raw.iloc[train_idx]
                    test_data = data_raw.iloc[test_idx]

                    if len(train_data) < model.seq_len or len(test_data) < model.seq_len:
                        logger.warning(f"Fold {fold + 1} của bảng {table_name} có dữ liệu quá ít để xử lý")
                        continue

                    # Graph and Embedding Layer
                    model.create_graph_layer(**graphParameter.creator.__dict__)
                    model.get_graph_layer().prepare_data(
                        train_data,
                        feature_cols=None,  # Sử dụng tất cả cột trừ timestamps và Label
                        timestamp_col="timestamps",
                        label_col="Label"
                    )
                    model.get_graph_layer().train(
                        epochs=graphParameter.training.epochs,
                        eval_every=graphParameter.training.eval_every
                    )

                    # Lấy embeddings từ GNN
                    data_classifier = model.gnn_system.get_embeddings(layer='last')
                    logger.info(f"Đã tạo embeddings từ GNN: {data_classifier.shape}")

                    # Classification Layer
                    model.create_gan_model(**classifierParameter.gan.__dict__)
                    model.create_rl_agent(**classifierParameter.rl.__dict__)

                    # Huấn luyện mô hình
                    model.fit(
                        data_classifier,
                        train_data['timestamps'],
                        train_data['Label'],
                        val_data=(test_data.drop(columns=['timestamps', 'Label']),
                                test_data['timestamps'],
                                test_data['Label'])
                    )
                    logger.info(f"Hoàn thành huấn luyện fold {fold + 1} cho bảng {table_name}")

                    # Kiểm tra trên tập test
                    test_predictions = model.predict(
                        test_data.drop(columns=['timestamps', 'Label']),
                        test_data['timestamps']
                    )
                    true_labels = test_data['Label'].values

                    # Đánh giá hiệu suất
                    accuracy = accuracy_score(true_labels[:len(test_predictions)], test_predictions)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        true_labels[:len(test_predictions)],
                        test_predictions,
                        average='weighted',
                        zero_division=0
                    )

                    # Lưu kết quả
                    results[f"{table_name}_fold_{fold + 1}"] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                    logger.info(f"Kết quả fold {fold + 1} cho bảng {table_name}: "
                                f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
                                f"Recall={recall:.4f}, F1={f1:.4f}")

            except Exception as e:
                logger.error(f"Lỗi khi xử lý bảng {table_name}: {str(e)}")
                continue

        # Tóm tắt kết quả
        if results:
            avg_metrics = {
                'accuracy': np.mean([r['accuracy'] for r in results.values()]),
                'precision': np.mean([r['precision'] for r in results.values()]),
                'recall': np.mean([r['recall'] for r in results.values()]),
                'f1': np.mean([r['f1'] for r in results.values()])
            }
            logger.info(f"Kết quả trung bình trên tất cả các fold và bảng: {avg_metrics}")
            return {
                "message": "Training and testing completed successfully.",
                "results": results,
                "average_metrics": avg_metrics
            }
        else:
            logger.warning("Không có kết quả nào được tạo ra do lỗi hoặc thiếu dữ liệu")
            return {"message": "No results generated due to errors or insufficient data."}

    except Exception as e:
        logger.error(f"Lỗi tổng quát trong quá trình huấn luyện và kiểm tra: {str(e)}")
        return {"message": f"Error during training and testing: {str(e)}"}