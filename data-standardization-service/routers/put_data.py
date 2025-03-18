from operator import index

from fastapi import APIRouter

from core.elastic import client

router = APIRouter()


@router.get("/training", tags=["put_data"])
async def put_data_training():
    query = {
        "query": {
            "match_all": {}
        }
    }
    indexs = client.get
    response = client.search(index="logs-bgl-*", body=query, size=1000)  # size là số lượng bản ghi cần lấy

    data = [hit["_source"] for hit in response["hits"]["hits"]]

    return data

@router.get("/testing", tags=["put_data"])
async def put_data_Testing():
    query = {
        "query": {
            "match_all": {}
        }
    }
    indexs = client.get
    response = client.search(index="logs-bgl-*", body=query, size=1000)

    data = [hit["_source"] for hit in response["hits"]["hits"]]

    return data