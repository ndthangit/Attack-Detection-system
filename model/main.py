from fastapi import FastAPI
import mlflow
# mlflow server  --backend-store-uri postgresql://postgres:postgres@localhost:5433/model  --artifacts-destination s3://bucket  --host 0.0.0.0  --port 5000
mlflow.autolog()
app = FastAPI()
# app.include_router(training_router, prefix="/data_train")
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/{index_name}")
async def fitting_model(index_name:str):

    return {"message": f"Hello {index_name}"}
