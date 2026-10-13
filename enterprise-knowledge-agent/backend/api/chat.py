"""聊天API - 处理用户对话请求"""

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse

from backend.models.schemas import ChatRequest, ConversationHistory
from backend.core.agent import EnterpriseAgent
from backend.api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["聊天"])

# 独立线程池：避免占用 FastAPI 默认的 ThreadPoolExecutor（默认仅 40 线程）
_stream_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="stream")

# Agent 实例池：{user_id: (agent, last_access_time)}
_agents: Dict[str, Tuple[EnterpriseAgent, float]] = {}
_agents_lock = threading.Lock()

# 配置：Agent 实例最大空闲时间（秒），超过后自动回收
_AGENT_TTL = 3600  # 1 小时
# 配置：最大同时在线 Agent 数量
_MAX_AGENTS = 50


def _close_agent(agent: EnterpriseAgent):
    """显式释放 Agent 资源（持久化记忆 + 关闭 ChromaDB 连接）"""
    try:
        # 持久化长期记忆（对话摘要、用户偏好等），防止 TTL 驱逐丢失
        if hasattr(agent, 'memory_manager') and agent.memory_manager:
            try:
                agent.memory_manager._save_long_term_memory()
            except Exception:
                pass
        if hasattr(agent, 'rag_engine') and agent.rag_engine:
            if hasattr(agent.rag_engine, '_vector_store') and agent.rag_engine._vector_store:
                try:
                    agent.rag_engine._vector_store._client.close()
                except Exception:
                    pass
    except Exception:
        pass


def _evict_expired_agents():
    """回收过期的 Agent 实例（在获取锁后调用）"""
    now = time.time()
    expired = [uid for uid, (_, ts) in _agents.items() if now - ts > _AGENT_TTL]
    for uid in expired:
        agent, _ = _agents.pop(uid)
        _close_agent(agent)
        logger.info(f"[AGENT] 回收过期实例: {uid}")


def get_agent(user_id: str = "default", web_search: bool = False) -> EnterpriseAgent:
    """获取或创建 Agent 实例（带 TTL 自动回收和容量限制）"""
    with _agents_lock:
        _evict_expired_agents()

        if user_id in _agents:
            agent, _ = _agents[user_id]
            _agents[user_id] = (agent, time.time())
            # 动态更新网络搜索状态
            agent.set_web_search(web_search)
            return agent

        # 容量限制：驱逐最久未访问的实例
        if len(_agents) >= _MAX_AGENTS:
            oldest_uid = min(_agents, key=lambda k: _agents[k][1])
            agent, _ = _agents.pop(oldest_uid)
            _close_agent(agent)
            logger.info(f"[AGENT] 驱逐最久未访问实例: {oldest_uid}")

        agent = EnterpriseAgent(user_id, web_search=web_search)
        _agents[user_id] = (agent, time.time())
        logger.info(f"[AGENT] 创建新实例: {user_id} (当前共 {len(_agents)} 个)")
        return agent


@router.post(
    "/stream",
    summary="流式对话",
    description="通过 SSE（Server-Sent Events）协议流式返回 Agent 回答。支持客户端断开检测和优雅取消。"
                "\n\n**SSE 事件类型：**"
                "\n- `sources`：参考来源和检索片段"
                "\n- `tool_call`：工具调用开始"
                "\n- `tool_result`：工具调用结果"
                "\n- `chunk`：回答文本片段"
                "\n- `evaluation`：质量评估结果（需开启 EVAL_ENABLED）"
                "\n- `[DONE]`：流式结束标记",
    responses={
        200: {"description": "SSE 流式响应", "content": {"text/event-stream": {}}},
        401: {"description": "未认证或 Token 过期"},
        500: {"description": "流式对话失败"},
    }
)
async def chat_stream(request: Request, chat_req: ChatRequest, user_id: str = Depends(get_current_user)):
    try:
        agent = get_agent(user_id, web_search=chat_req.web_search)
        logger.info(f"[CHAT] user={user_id}, web_search={chat_req.web_search}")

        q = asyncio.Queue()
        stop_event = threading.Event()  # 跨线程取消信号

        def _run_sync():
            """在线程中运行同步生成器，支持取消"""
            try:
                for item in agent.chat_stream(
                    chat_req.message,
                    should_stop=lambda: stop_event.is_set()
                ):
                    if stop_event.is_set():
                        logger.info(f"[CHAT] 检测到停止信号，中断生成: {user_id}")
                        break
                    q.put_nowait(item)
            except Exception as e:
                if not stop_event.is_set():
                    q.put_nowait({"type": "chunk", "data": f"\n\n❌ 错误: {str(e)}"})
            finally:
                q.put_nowait(None)  # 结束标记

        # 在独立线程池中启动同步生成器（避免占用 FastAPI 默认线程池）
        loop = asyncio.get_running_loop()
        loop.run_in_executor(_stream_executor, _run_sync)

        async def generate():
            """异步生成器：检测客户端断开并通知后台线程

            注意：yield 不能放在 finally 块中（Python 限制），所以每个退出路径
            都需要手动设置 stop_event 并 yield [DONE]。
            """
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            logger.info(f"[CHAT] 客户端断开连接: {user_id}")
                            stop_event.set()
                            yield "data: [DONE]\n\n"
                            return
                        continue

                    if item is None:
                        # 正常结束
                        yield "data: [DONE]\n\n"
                        return
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                logger.info(f"[CHAT] 异步任务被取消（客户端断开）: {user_id}")
                stop_event.set()
                # 不 yield，直接让协程终止
                raise
            except Exception as e:
                logger.error(f"[CHAT] generate 异常: {e}")
                stop_event.set()
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"流式对话失败: {str(e)}")


@router.get(
    "/history",
    response_model=ConversationHistory,
    summary="获取对话历史",
    description="获取当前用户的对话历史记录，按时间正序返回。",
    responses={
        200: {"description": "返回对话历史"},
        401: {"description": "未认证或 Token 过期"},
    }
)
async def get_conversation_history(max_messages: int = 20, user_id: str = Depends(get_current_user)):
    try:
        agent = get_agent(user_id)
        history = agent.memory_manager.get_conversation_history(max_messages)

        return ConversationHistory(
            messages=history,
            total_count=len(history)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取对话历史失败: {str(e)}")


@router.delete(
    "/history",
    summary="清空对话历史",
    description="清空当前用户的所有对话历史和对话摘要。清空后不可恢复。",
    responses={
        200: {"description": "清空成功"},
        401: {"description": "未认证或 Token 过期"},
        500: {"description": "清空失败"},
    }
)
async def clear_conversation_history(user_id: str = Depends(get_current_user)):
    try:
        agent = get_agent(user_id)
        agent.clear_conversation()
        return {"message": "对话历史已清空", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空对话历史失败: {str(e)}")


@router.get(
    "/agents",
    summary="列出 Agent 实例",
    description="列出当前所有活跃的 Agent 实例（调试用）。每个用户有独立的 Agent 实例，TTL 1小时自动回收。",
    responses={
        200: {"description": "返回活跃 Agent 列表"},
        401: {"description": "未认证或 Token 过期"},
    }
)
async def list_agents(user_id: str = Depends(get_current_user)):
    with _agents_lock:
        return {
            "agents": list(_agents.keys()),
            "count": len(_agents)
        }
