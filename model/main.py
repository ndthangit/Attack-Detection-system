from fastapi import FastAPI
import routers
app = FastAPI()
app.include_router(routers.training_router, prefix="/training")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/upload")
async def upload_data():
    return {"message": f"Hello "}
