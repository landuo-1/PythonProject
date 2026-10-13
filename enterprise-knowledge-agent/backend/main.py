"""FastAPI主应用入口"""

import logging
import os
import sys
import threading
from pathlib import Path

# 统一日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager

from backend.api.chat import router as chat_router
from backend.api.documents import router as documents_router
from backend.api.knowledge_base import router as knowledge_base_router
from backend.api.auth import router as auth_router
from backend.config.settings import settings

# 全局预加载状态
_preload_done = threading.Event()


def _preload_modules():
    """后台线程预加载耗时模块"""
    try:
        logger.info("[INIT] 后台预加载核心模块...")
        from backend.core.rag import RAGEngine
        from backend.core.llm import get_mimo_llm

        _ = RAGEngine()      # embedding 模型加载（~1s）
        _ = get_mimo_llm()   # langchain_openai 导入（~6s）

        # 如果开启了 Reranker，后台预加载模型（避免首次请求阻塞）
        if settings.RERANKER_ENABLED:
            try:
                logger.info("[INIT] 预加载 Reranker 模型...")
                from sentence_transformers import CrossEncoder
                CrossEncoder('BAAI/bge-reranker-base')
                logger.info("[INIT] Reranker 模型加载完成")
            except Exception as e:
                logger.error(f"[INIT] Reranker 预加载失败: {e}")

        logger.info("[INIT] 核心模块加载完成")
    except Exception as e:
        logger.error(f"[INIT] 预加载失败: {e}")
    finally:
        _preload_done.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("[STARTUP] 企业智能知识库Agent启动中...")

    # 安全检查：JWT Secret Key
    if settings.JWT_SECRET_KEY in ("dev-secret-key-change-in-production", "change-me", ""):
        logger.warning("[SECURITY] ⚠️ JWT_SECRET_KEY 使用了不安全的默认值，生产环境请务必修改！")

    # 创建必要的目录
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    memory_dir = str(Path(settings.CHROMA_PERSIST_DIR).parent / "memory")
    os.makedirs(memory_dir, exist_ok=True)

    # 后台线程预加载（不阻塞服务器启动）
    threading.Thread(target=_preload_modules, daemon=True).start()

    logger.info("[OK] 服务器已就绪（核心模块后台加载中）")
    logger.info(f"[INFO] 上传目录: {settings.UPLOAD_DIR}")
    logger.info(f"[INFO] 向量数据库: {settings.CHROMA_PERSIST_DIR}")
    logger.info(f"[INFO] LLM模型: {settings.MIMO_MODEL}")

    yield

    logger.info("[SHUTDOWN] 企业智能知识库Agent关闭")


# 配置 Bearer Token 安全方案（让 Swagger Authorize 按钮生效）
security_scheme = HTTPBearer(
    scheme_name="Bearer",
    description="输入登录接口返回的 token（不需要加 Bearer 前缀，Swagger 会自动添加）"
)

# 创建FastAPI应用
app = FastAPI(
    title="企业智能知识库Agent API",
    description=(
        "基于 LangGraph 自纠错工作流 + RAG 混合检索的企业知识库问答系统\n\n"
        "## 主要功能\n"
        "- **智能问答**：基于企业文档的 RAG 检索问答\n"
        "- **混合检索**：向量检索 + BM25 关键词检索 + RRF 融合排序\n"
        "- **质量评估**：忠实度 + 答案相关性，不达标自动重检索重生成\n"
        "- **多用户隔离**：JWT 认证 + 个人工作区/团队共享库\n"
        "- **流式对话**：SSE 流式推送，支持客户端断开检测\n\n"
        "## 认证方式\n"
        "使用 `/api/auth/login` 获取 JWT Token，然后点击右上角 **Authorize** 按钮输入 Token。\n"
        "如果 Token 无效或过期，所有需要认证的接口都会返回 401 。"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "认证", "description": "用户注册和登录"},
        {"name": "聊天", "description": "流式对话和对话历史管理"},
        {"name": "文档管理", "description": "文档上传、列表、删除和内容预览"},
        {"name": "知识库管理", "description": "系统状态、知识库信息和清空操作"},
    ]
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit 默认端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(knowledge_base_router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "企业智能知识库Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "ready": _preload_done.is_set(),
        "components": {
            "api": "running",
            "llm": settings.MIMO_MODEL,
            "vector_db": "ChromaDB"
        }
    }


# 全局异常处理器（统一错误响应格式）
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """HTTP 异常统一处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.status_code, "message": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """未捕获异常兜底处理"""
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "服务器内部错误，请稍后重试"},
    )


if __name__ == "__main__":
    import uvicorn

    # 开发模式：reload=True 改代码后自动重启（不需要手动 Ctrl+C）
    # 生产模式：改为 reload=False
    uvicorn.run(
        "main:app",  # 字符串形式，reload 模式必须用字符串
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )
