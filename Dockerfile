FROM python:3.11-slim

WORKDIR /app

# 升级 pip 避免旧版本安装失败
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# 使用 HTTP 服务器模式（支持 K8s 健康检查）
CMD ["python", "server.py"]
