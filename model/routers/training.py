from operator import index
from fastapi import APIRouter
from sqlalchemy import inspect

from core.config import GraphConfig, ClassifierParameters
from core.database import engine
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from purposed_model.CustomPWAGAT import CustomPWAGAT

router = APIRouter()

@router.get("/", tags=["Experiments"])
async def training():
    # Khởi tạo các tham số cho mô hình

    model = CustomPWAGAT()

    # Get all table names in the database
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    print(table_names)

    for table_name in table_names:
        print(f"Processing table: {table_name}")
        try:
            # Load table data into a Pandas DataFrame

            # graph layer
            query = f'SELECT * FROM "{table_name}";'
            data_raw = pd.read_sql(query, engine)

            'graph and embedding layer'

            graphParameter = GraphConfig()
            model.create_graph_layer(**graphParameter.creator.__dict__)
            model.get_graph_layer().prepare_data(
                data_raw,
                feature_cols=None,
                time_series_cols="timestamp",
                label_col="Label"
            )
            model.get_graph_layer().train(
                epochs=graphParameter.training.epochs,
                eval_every=graphParameter.training.eval_every
            )

            data_classifier = model.gnn_system.get_embeddings(layer='last')

            'GAN model'

            classifierParameter = ClassifierParameters()

            model.create_gan_model(**classifierParameter.gan.__dict__)
            model.create_rl_agent(**classifierParameter.rl.__dict__)
            model.fit(data_classifier,data_raw['timestamps'],data_raw['Label'])

            # print(data_raw.columns)
        except Exception as e:
            print(f"Error processing table {table_name}: {str(e)}")


    return {"message": "Training completed successfully."}
