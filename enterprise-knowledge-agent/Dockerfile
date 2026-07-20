# 使用Python 3.10基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY conftest.py ./

# 创建 .env 占位（运行时通过 -v 或 --env-file 覆盖）
RUN touch .env

# 创建数据目录
RUN mkdir -p data/uploads data/chroma_db data/memory

# 暴露端口
EXPOSE 8000 8501

# 启动脚本
COPY start.sh .
RUN chmod +x start.sh

# 启动应用
CMD ["./start.sh"]
