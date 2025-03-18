from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.elastic import client

from routers.user import router as user_router
from routers.upload import router as upload_router
from routers.put_data import router as put_router
app = FastAPI()
app.include_router(user_router, prefix="/api")
app.include_router(upload_router, prefix="/upload")
app.include_router(put_router, prefix="/put_data")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
def check_elasticsearch():
    """Kiểm tra kết nối Elasticsearch"""
    return {"status": "connected"} if client.ping() else {"status": "failed"}
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
