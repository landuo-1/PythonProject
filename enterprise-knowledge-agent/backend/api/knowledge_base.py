"""知识库API - 管理知识库"""

from fastapi import APIRouter, HTTPException, Depends

from backend.models.schemas import KnowledgeBaseInfo, SystemStatus
from backend.api.auth import get_current_user

router = APIRouter(prefix="/api/knowledge-base", tags=["知识库管理"])

# Agent 实例管理（与chat模块共享）
from backend.api.chat import get_agent


@router.get("/system/status", response_model=SystemStatus)
async def get_system_status():
    """获取系统状态（轻量级检查，不创建重型实例）"""
    try:
        components = {}

        # 检查 LLM 模块是否可导入
        try:
            from backend.core.llm import get_mimo_llm
            components["llm"] = "正常"
        except Exception as e:
            components["llm"] = f"异常: {str(e)}"

        # 检查 RAG 模块是否可导入
        try:
            from backend.core.rag import RAGEngine
            components["rag"] = "正常"
        except Exception as e:
            components["rag"] = f"异常: {str(e)}"

        # 检查记忆模块是否可导入
        try:
            from backend.core.memory import MemoryManager
            components["memory"] = "正常"
        except Exception as e:
            components["memory"] = f"异常: {str(e)}"

        # 检查工作流模块是否可导入
        try:
            from backend.core.agent import EnterpriseAgent
            components["workflow"] = "正常"
        except Exception as e:
            components["workflow"] = f"异常: {str(e)}"

        status = "healthy" if all("正常" in v for v in components.values()) else "degraded"

        return SystemStatus(
            status=status,
            version="1.0.0",
            components=components
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


@router.delete("/my-documents")
async def clear_my_documents(user_id: str = Depends(get_current_user)):
    """清空当前用户的所有文档（需要登录）"""
    try:
        agent = get_agent(user_id)
        deleted_count = agent.rag_engine.clear_user_documents(user_id)
        # 同步清理对话摘要，防止 LLM 从历史中引用已删除的文档
        agent.memory_manager.update_conversation_summary("")

        return {
            "message": f"已清空 {deleted_count} 个文档片段",
            "deleted_count": deleted_count,
            "user_id": user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空文档失败: {str(e)}")


@router.get("/my-info", response_model=KnowledgeBaseInfo)
async def get_my_knowledge_base_info(user_id: str = Depends(get_current_user)):
    """获取当前用户的知识库信息（需要登录）"""
    try:
        agent = get_agent(user_id)
        info = agent.get_knowledge_base_info()

        return KnowledgeBaseInfo(
            document_count=info["document_count"],
            memory_stats=info["memory_stats"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取知识库信息失败: {str(e)}")
