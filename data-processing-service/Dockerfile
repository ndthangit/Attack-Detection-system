# Sử dụng image Python chính thức làm base image
FROM python:3.12.9

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các dependencies cần thiết
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Mở cổng mà FastAPI sẽ chạy
EXPOSE 8001

# Chạy ứng dụng FastAPI với Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]