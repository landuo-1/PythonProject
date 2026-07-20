#!/bin/bash

# 启动脚本 - 同时启动后端和前端服务

echo "[STARTUP] 启动企业智能知识库Agent..."

# 创建数据目录
mkdir -p data/uploads data/chroma_db data/memory

# 在项目根目录启动后端服务
echo "[INFO] 启动后端服务 (FastAPI)..."
python -m backend.main &
BACKEND_PID=$!

# 等待后端启动
echo "[INFO] 等待后端服务启动..."
sleep 5

# 检查后端是否启动成功
if curl -s http://localhost:8000/health > /dev/null; then
    echo "[OK] 后端服务启动成功"
else
    echo "[ERROR] 后端服务启动失败"
    exit 1
fi

# 启动前端服务
echo "[INFO] 启动前端服务 (Streamlit)..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!

echo "[OK] 服务启动完成"
echo "[INFO] 后端API: http://localhost:8000"
echo "[INFO] API文档: http://localhost:8000/docs"
echo "[INFO] 前端界面: http://localhost:8501"

# 等待进程
wait $BACKEND_PID $FRONTEND_PID
