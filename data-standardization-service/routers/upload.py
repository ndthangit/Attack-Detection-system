from models.index_template import Template

from fastapi import APIRouter, UploadFile, File
from models.index.create_index import create_indices
from core.elastic import client
router = APIRouter()


@router.post("/", tags=["upload"])
async def upload_new_data(file: UploadFile = File(...)):
    contents = await file.read()

    print(contents)  # Print the file contents received from the frontend
    return {"filename": file.filename, "content_type": file.content_type}


@router.post("/create-indices", tags=["upload"])
async def create_indices_endpoint(template: Template):
    create_indices(template)
    return {"message": "Index created successfully"}