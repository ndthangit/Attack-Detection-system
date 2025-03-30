FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /usr/src/app

COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

# Mở cổng mà FastAPI sẽ chạy
EXPOSE 8001

# Chạy ứng dụng FastAPI với Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]