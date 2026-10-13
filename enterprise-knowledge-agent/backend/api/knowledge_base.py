"""知识库API - 管理知识库"""

from fastapi import APIRouter, HTTPException, Depends

from backend.models.schemas import KnowledgeBaseInfo, SystemStatus
from backend.api.auth import get_current_user

router = APIRouter(prefix="/api/knowledge-base", tags=["知识库管理"])

# Agent 实例管理（与chat模块共享）
from backend.api.chat import get_agent


@router.get(
    "/system/status",
    response_model=SystemStatus,
    summary="系统状态",
    description="获取系统各组件的运行状态（轻量级检查，不创建重型实例）。"
                "\n\n**检查组件：** LLM、RAG、Memory、Workflow",
    responses={
        200: {"description": "返回系统状态"},
        500: {"description": "获取状态失败"},
    }
)
async def get_system_status():
    try:
        components = {}

        try:
            from backend.core.llm import get_mimo_llm
            components["llm"] = "正常"
        except Exception as e:
            components["llm"] = f"异常: {str(e)}"

        try:
            from backend.core.rag import RAGEngine
            components["rag"] = "正常"
        except Exception as e:
            components["rag"] = f"异常: {str(e)}"

        try:
            from backend.core.memory import MemoryManager
            components["memory"] = "正常"
        except Exception as e:
            components["memory"] = f"异常: {str(e)}"

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


@router.delete(
    "/my-documents",
    summary="清空我的文档",
    description="清空当前用户上传的所有文档（个人工作区 + 团队共享库）。"
                "\n\n**注意：** 此操作不可恢复，会同时清理向量库、BM25 索引和磁盘文件。",
    responses={
        200: {"description": "清空成功，返回删除的片段数"},
        401: {"description": "未认证或 Token 过期"},
        500: {"description": "清空失败"},
    }
)
async def clear_my_documents(user_id: str = Depends(get_current_user)):
    try:
        agent = get_agent(user_id)
        deleted_count = agent.rag_engine.clear_user_documents(user_id)
        agent.memory_manager.update_conversation_summary("")

        return {
            "message": f"已清空 {deleted_count} 个文档片段",
            "deleted_count": deleted_count,
            "user_id": user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空文档失败: {str(e)}")


@router.get(
    "/my-info",
    response_model=KnowledgeBaseInfo,
    summary="知识库信息",
    description="获取当前用户的知识库统计信息，包括文档数量和记忆统计。",
    responses={
        200: {"description": "返回知识库信息"},
        401: {"description": "未认证或 Token 过期"},
        500: {"description": "获取信息失败"},
    }
)
async def get_my_knowledge_base_info(user_id: str = Depends(get_current_user)):
    try:
        agent = get_agent(user_id)
        info = agent.get_knowledge_base_info()

        return KnowledgeBaseInfo(
            document_count=info["document_count"],
            memory_stats=info["memory_stats"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取知识库信息失败: {str(e)}")
