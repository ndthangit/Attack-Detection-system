from fastapi import FastAPI
from core.elastic import client
from core.database import engine
import export_sentence.export_save
import export_sentence

app = FastAPI()

@app.get("/")
async def root():
    num_sample=export_sentence.export_save.export_feature_save("aminer-fox")
    return { f"num_sample: {num_sample}"}

@app.get("/get_data")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/get_extracted_data")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}