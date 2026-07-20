from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache
from pathlib import Path

# 获取 backend 目录（所有数据统一存放在 backend/data/ 下）
backend_dir = Path(__file__).parent.parent
project_root = backend_dir.parent
env_file_path = project_root / ".env"
data_dir = backend_dir / "data"

class Settings(BaseSettings):
    """应用配置"""

    # MIMO LLM配置
    MIMO_API_KEY: str
    MIMO_BASE_URL: str = "https://token-plan-cn.xiaomimimo.com/v1"
    MIMO_MODEL: str = "mimo-v2.5"

    # ChromaDB配置
    CHROMA_PERSIST_DIR: str = str(data_dir / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "enterprise_knowledge"

    # 检索配置
    RERANKER_ENABLED: bool = False  # 开启重排序（首次需下载模型约 1.3GB）

    # 文档处理配置
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # JWT 认证配置（必填）
    JWT_SECRET_KEY: str

    # Tavily搜索配置（可选）
    TAVILY_API_KEY: Optional[str] = None

    # 质量评估配置
    EVAL_ENABLED: bool = False  # 是否启用质量评估（规则检查，无额外模型）
    EVAL_FAITHFULNESS_THRESHOLD: float = 0.90  # 忠实度阈值
    EVAL_RELEVANCY_THRESHOLD: float = 0.75  # 答案相关性阈值
    EVAL_DEGRADATION_STRATEGY: str = "retry"  # 降级策略: retry / fallback / human_review

    # 文件上传配置
    UPLOAD_DIR: str = str(data_dir / "uploads")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list = [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls"]

    class Config:
        env_file = str(env_file_path)
        env_file_encoding = "utf-8"
        extra = "ignore"  # 忽略 .env 中的多余变量（如旧版 BM25_WEIGHT）

@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()

settings = get_settings()
