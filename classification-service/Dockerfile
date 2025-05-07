FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10
#FROM python:3.9-slim

WORKDIR /usr/src/app

COPY ./requirements.txt /usr/src/app/requirements.txt
# Nâng cấp pip trước
RUN python -m pip install --upgrade pip

# Cài psycopg2 riêng để tránh lỗi biên dịch
RUN pip install --no-cache-dir psycopg2-binary

RUN pip install --no-cache-dir --timeout=100 -r requirements.txt

COPY . /usr/src/app

# Mở cổng mà FastAPI sẽ chạy
EXPOSE 8001

# Chạy ứng dụng FastAPI với Uvicorn4
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]