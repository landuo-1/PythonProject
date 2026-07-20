"""Agent核心模块 - 基于 LangGraph 的完整自纠错工作流

工作流状态图（internal 路由）：
  START → router → retrieve → generate → evaluate
                                        ├─ pass → END
                                        └─ fail → retry_retrieve → generate → evaluate → END
"""

import json
import logging
import re
import threading
from typing import Callable, Dict, Any, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from backend.core.llm import get_mimo_llm_for_agent
from backend.core.rag import RAGEngine
from backend.core.memory import MemoryManager
from backend.config.settings import settings

logger = logging.getLogger(__name__)


# ==================== 状态定义 ====================

class AgentState(TypedDict):
    """工作流状态（仅包含可序列化的数据）"""
    message: str
    user_id: str
    route: str
    search_query: str       # 检索查询（首次=HyDE改写，重试=增强查询）
    context: str
    sources: list
    chunks: list
    search_result: str
    prompt: str
    response: str           # LLM 生成的回答
    eval_result: dict       # 评估结果
    attempt: int            # 当前尝试次数
    needs_retry: bool       # 是否需要重试
    web_search_enabled: bool


# ==================== 意图路由器 ====================

class IntentRouter:
    """意图路由器 - 使用 LLM 分类"""

    EXTERNAL_KEYWORDS = [
        '天气', '气温', '下雨', '下雪',
        '新闻', '最新消息', '热点', '头条',
        '股价', '汇率', '基金', '比特币',
    ]

    CLASSIFICATION_PROMPT = """你是一个意图分类器。根据用户的问题，判断应该使用哪个路由。

路由选项：
- internal: 企业内部信息（公司政策、产品、制度、规章、内部流程等）
- external: 需要外部实时信息（天气、新闻、股价、汇率、最新技术动态等）
- internal_first: 可能有内部信息但不确定，优先内部无结果再查外部
- general: 简单问候、闲聊

分类规则：
1. 明确是企业内部问题 → internal
2. 明确需要实时外部信息 → external
3. 不确定是否有内部信息 → internal_first
4. 简单问候、闲聊 → general

只输出一个词：internal、external、internal_first 或 general。不要输出其他内容。

用户问题：{message}
路由："""

    @classmethod
    def classify(cls, message: str, llm=None, web_search_enabled: bool = False) -> str:
        msg = message.strip()

        if re.match(r'^(你好|hi|hello|嗨|hey|谢谢|thanks|再见|bye)$', msg, re.IGNORECASE):
            return "general"

        if any(kw in msg for kw in cls.EXTERNAL_KEYWORDS):
            if web_search_enabled:
                logger.info(f"[Router] 关键词匹配: {msg[:20]}... → external")
                return "external"
            else:
                logger.info(f"[Router] 关键词匹配但搜索未启用: {msg[:20]}... → external_disabled")
                return "external_disabled"

        if re.search(r'(搜索|查找|搜一下|找一下|帮我查|帮我搜|网上查)', msg):
            logger.info(f"[Router] 搜索请求: {msg[:20]}... → internal_first")
            return "internal_first"

        if llm is not None:
            try:
                from langchain_core.messages import HumanMessage
                prompt = cls.CLASSIFICATION_PROMPT.format(message=msg)
                response = llm.invoke([HumanMessage(content=prompt)])
                route = response.content.strip().lower()

                for r in ("internal_first", "internal", "external", "general"):
                    if r in route:
                        if r == "external" and not web_search_enabled:
                            logger.info(f"[Router] LLM判定外部但搜索未启用: {msg[:20]}... → external_disabled")
                            return "external_disabled"
                        logger.info(f"[Router] LLM分类: {msg[:20]}... → {r}")
                        return r

                logger.warning(f"[Router] LLM返回无效: {route}, 降级到规则分类")
                return cls._rule_classify(msg)
            except Exception as e:
                logger.error(f"[Router] LLM分类失败: {e}, 降级到规则分类")
                return cls._rule_classify(msg)

        return "internal"

    @classmethod
    def _rule_classify(cls, msg: str) -> str:
        if any(kw in msg for kw in cls.EXTERNAL_KEYWORDS):
            return "external"
        if re.search(r'(搜索|查找|搜一下|找一下|帮我查|帮我搜|网上查)', msg):
            return "internal_first"
        if re.match(r'^(你好|hi|hello|谢谢|再见)$', msg, re.IGNORECASE):
            return "general"
        return "internal"


# ==================== 工作流节点 ====================

def _create_node_fns(llm, rag_engine: RAGEngine, memory_manager: MemoryManager):
    """创建工作流节点函数（通过闭包捕获 llm、rag、memory）"""

    def router_node(state: AgentState) -> dict:
        message = state["message"]
        web_search = state.get("web_search_enabled", False)
        route = IntentRouter.classify(message, llm, web_search)
        logger.info(f"[Router] user={state.get('user_id')}, route={route}, msg={message[:30]}")
        return {"route": route}

    def general_node(state: AgentState) -> dict:
        """通用回复 - 在工作流内直接生成"""
        from langchain_core.messages import HumanMessage, SystemMessage
        message = state["message"]
        history = memory_manager.get_langchain_messages(max_messages=6)
        messages = [SystemMessage(content="你是一个友好的企业知识库助手。请简洁、友好地回复。")] + history + [HumanMessage(content=message)]
        response = llm.invoke(messages)
        content = response.content or "你好！"
        return {"response": content, "sources": [], "chunks": []}

    def retrieve_node(state: AgentState) -> dict:
        """知识库检索（首次检索或重试检索）"""
        message = state["message"]
        user_id = state["user_id"]
        attempt = state.get("attempt", 0)

        if attempt == 0:
            # 首次检索：使用 HyDE 改写
            search_query = _hyde_rewrite(message, llm)
        else:
            # 重试检索：使用增强查询
            search_query = _enhance_query(message, state.get("response", ""), llm)

        rag_result = rag_engine.get_context_with_metadata(search_query, top_k=5, user_id=user_id, scope="all")
        context = rag_result["context"]
        sources = rag_result["sources"]
        chunks = rag_result["chunks"]

        prompt = _build_internal_prompt(context, memory_manager)
        logger.info(f"[Retrieve] attempt={attempt}, query={search_query[:30]}..., chunks={len(chunks)}")
        return {"search_query": search_query, "prompt": prompt, "context": context, "sources": sources, "chunks": chunks}

    def generate_node(state: AgentState) -> dict:
        """LLM 生成回答"""
        from langchain_core.messages import HumanMessage, SystemMessage
        message = state["message"]
        route = state.get("route", "internal")
        prompt = state.get("prompt", "")
        context = state.get("context", "")

        if route == "general":
            system_prompt = "你是一个友好的企业知识库助手。请简洁、友好地回复。"
        else:
            system_prompt = prompt

        if not system_prompt:
            return {"response": ""}

        history = memory_manager.get_langchain_messages(max_messages=6)
        if route in ("internal", "internal_first") and not context:
            history = _filter_history_for_empty_kb(history)

        messages = [SystemMessage(content=system_prompt)] + history + [HumanMessage(content=message)]
        response = llm.invoke(messages)
        content = _clean_response(response.content or "抱歉，我无法处理您的请求。")

        logger.info(f"[Generate] route={route}, response_len={len(content)}")
        return {"response": content}

    def evaluate_node(state: AgentState) -> dict:
        """质量评估（忠实度 + 答案相关性）"""
        if not settings.EVAL_ENABLED:
            return {"eval_result": None, "needs_retry": False}

        message = state["message"]
        response = state.get("response", "")
        chunks = state.get("chunks", [])
        attempt = state.get("attempt", 0)
        route = state.get("route", "internal")

        # 仅对 internal/internal_first 路由评估
        if route not in ("internal", "internal_first") or not response or not chunks:
            return {"eval_result": None, "needs_retry": False}

        try:
            from backend.core.evaluation import QualityEvaluator
            evaluator = QualityEvaluator.get_instance()

            context_parts = [c.get("content", "")[:300] for c in chunks[:3]]
            context = "\n".join(context_parts)

            eval_result = evaluator.evaluate(
                question=message,
                answer=response,
                context=context,
                chunks=chunks,
                faithfulness_threshold=settings.EVAL_FAITHFULNESS_THRESHOLD,
                relevancy_threshold=settings.EVAL_RELEVANCY_THRESHOLD,
            )

            needs_retry = (not eval_result["overall_passed"]
                          and attempt < 1
                          and settings.EVAL_DEGRADATION_STRATEGY == "retry")

            if needs_retry:
                logger.warning(f"[Evaluate] attempt={attempt}, 未达标，触发重检索重生成")
            else:
                logger.info(f"[Evaluate] attempt={attempt}, passed={eval_result['overall_passed']}")

            return {"eval_result": eval_result, "needs_retry": needs_retry, "attempt": attempt + 1}

        except Exception as e:
            logger.error(f"[Evaluate] 评估异常: {e}")
            return {"eval_result": None, "needs_retry": False}

    def external_node(state: AgentState) -> dict:
        """外部网络搜索 + 生成"""
        from langchain_core.messages import HumanMessage, SystemMessage
        message = state["message"]
        if not state.get("web_search_enabled", False):
            return {"response": "", "sources": [], "chunks": [], "search_result": ""}

        search_result = _execute_web_search(message)
        prompt = _build_external_prompt(search_result)
        history = memory_manager.get_langchain_messages(max_messages=6)
        messages = [SystemMessage(content=prompt)] + history + [HumanMessage(content=message)]
        response = llm.invoke(messages)
        content = _clean_response(response.content or "抱歉，我无法处理您的请求。")

        tool_calls = [{"tool": "search_web", "args": json.dumps({"query": message}, ensure_ascii=False), "result": search_result[:200]}]
        return {"response": content, "sources": [], "chunks": [], "search_result": search_result, "tool_calls": tool_calls}

    def external_disabled_node(state: AgentState) -> dict:
        """网络搜索未启用"""
        return {"response": "", "sources": [], "chunks": [], "search_result": "__disabled__"}

    return (router_node, general_node, retrieve_node, generate_node,
            evaluate_node, external_node, external_disabled_node)


def create_agent_workflow(llm, rag_engine: RAGEngine, memory_manager: MemoryManager):
    """
    创建 LangGraph 完整自纠错工作流

    工作流图：
    START → router → [general / external / external_disabled] → END
                    └→ retrieve → generate → evaluate
                                            ├─ pass → END
                                            └─ fail → END（降级处理由 chat_stream 负责）
    """
    (router_node, general_node, retrieve_node, generate_node,
     evaluate_node, external_node, external_disabled_node) = _create_node_fns(llm, rag_engine, memory_manager)

    def route_from_router(state: AgentState) -> str:
        route = state.get("route", "internal")
        if route == "general":
            return "general"
        elif route == "external":
            return "external"
        elif route == "external_disabled":
            return "external_disabled"
        else:
            return "internal"

    def after_evaluate(state: AgentState) -> str:
        """评估后决定：通过→END，需要重试→retrieve"""
        needs_retry = state.get("needs_retry", False)
        if needs_retry:
            return "retrieve"
        return "end"

    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("general", general_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("external", external_node)
    workflow.add_node("external_disabled", external_disabled_node)

    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        route_from_router,
        {
            "general": "general",
            "internal": "retrieve",
            "external": "external",
            "external_disabled": "external_disabled",
        }
    )

    # general / external / external_disabled → END
    workflow.add_edge("general", END)
    workflow.add_edge("external", END)
    workflow.add_edge("external_disabled", END)

    # 内部知识库：retrieve → generate → evaluate → [pass→END / retry→retrieve]
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        after_evaluate,
        {"end": END, "retrieve": "retrieve"}
    )

    return workflow.compile()


# ==================== 共用工具函数 ====================

def _build_internal_prompt(context: str, memory) -> str:
    base = """你是企业知识库助手。回答要完整、详细，用分点格式组织。

回答要求：
- 只基于知识库检索结果中实际引用的内容回答，不要编造
- 引用来源时，必须使用检索结果中给出的精确格式：📚 来源：《文件名——片段N》
- 片段编号N必须是检索结果中实际存在的编号，不要编造不存在的片段号
- 只标注你实际使用的来源，不要列出所有检索结果
- 不要输出 JSON、工具调用代码或技术格式的内容
- 如果知识库中没有相关信息，如实告知用户"""

    if context:
        base += f"\n\n## 知识库检索结果\n{context}\n\n请基于以上内容回答。引用来源时使用精确的片段编号。"
    else:
        base += "\n\n当前知识库为空或未找到相关信息。请直接告知用户：知识库中暂无相关内容，建议上传相关文档后再提问。"

    memory_context = memory.get_memory_context_for_prompt()
    if memory_context:
        base += f"\n\n## 记忆上下文\n{memory_context}"
    return base


def _build_external_prompt(search_result: str) -> str:
    return f"""你是企业知识助手。用户询问了需要网络搜索的问题。

## 搜索结果
{search_result}

## 回答要求
- 基于搜索结果回答，用分点格式组织
- 标注信息来源（搜索结果中的链接）
- 不要输出 JSON 或技术格式的内容
- 以下为外部网络信息，仅供参考"""


def _execute_web_search(query: str) -> str:
    try:
        from backend.core.tools import search_web
        return search_web(query)
    except Exception as e:
        logger.error(f"网络搜索失败: {e}")
        return f"网络搜索失败: {str(e)}"


def _hyde_rewrite(message: str, llm) -> str:
    """HyDE 查询改写：对短查询或模糊查询生成假设性文档"""
    needs_rewrite = (
        len(message) < 8
        or any(kw in message for kw in ['那个', '这个', '什么', '哪個', '哪个'])
    )

    if not needs_rewrite:
        return message

    try:
        from langchain_core.messages import HumanMessage
        hyde_prompt = f"请根据以下用户问题，生成一段可能存在于企业文档中的完整陈述句（不要回答问题，只生成陈述句）：\n\n{message}"
        response = llm.invoke([HumanMessage(content=hyde_prompt)])
        hypothetical = response.content.strip()
        if hypothetical and len(hypothetical) > len(message):
            logger.info(f"[HyDE] 改写: {message[:20]}... → {hypothetical[:30]}...")
            return hypothetical
    except Exception as e:
        logger.warning(f"[HyDE] 改写失败: {e}")

    return message


def _enhance_query(original_query: str, failed_response: str, llm) -> str:
    """
    增强查询：当评估不达标时，基于原始查询和失败回答生成更精确的检索查询

    Args:
        original_query: 用户原始问题
        failed_response: 未通过评估的 LLM 回答
        llm: LLM 实例

    Returns:
        增强后的检索查询
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        prompt = f"""用户问题：{original_query}

之前的回答未能通过质量评估。请根据用户问题，生成一个更精确的检索查询（用于在企业知识库中搜索相关文档）。
要求：
- 只输出查询文本，不要解释
- 查询应该包含关键术语，提高检索精度
- 不要重复用户问题的模糊表述"""
        response = llm.invoke([SystemMessage(content="你是一个检索查询优化器。"), HumanMessage(content=prompt)])
        enhanced = response.content.strip()
        if enhanced and enhanced != original_query:
            logger.info(f"[Enhance] 查询增强: {original_query[:20]}... → {enhanced[:30]}...")
            return enhanced
    except Exception as e:
        logger.warning(f"[Enhance] 查询增强失败: {e}")

    return original_query


def _filter_history_for_empty_kb(history_messages: list) -> list:
    """过滤历史中知识库相关的消息（当知识库为空时使用）"""
    from langchain_core.messages import AIMessage, HumanMessage

    if not history_messages:
        return history_messages

    filtered = []
    skip_next = False
    for msg in history_messages:
        if skip_next:
            skip_next = False
            continue

        content = msg.content if msg.content else ""

        if isinstance(msg, AIMessage) and "来源" in content and ("《" in content or "片段" in content):
            skip_next = True
            continue

        filtered.append(msg)

    if len(history_messages) != len(filtered) and len(filtered) > 2:
        return filtered[-2:]
    return filtered


def _filter_cited_sources(response: str, sources: list, chunks: list) -> tuple:
    """根据 LLM 回答内容，精确过滤出被引用的来源和片段"""
    if not response or not sources:
        return sources, chunks

    chunk_map = {}
    for c in chunks:
        fname = c.get("filename", "")
        idx = c.get("index", 0)
        if fname:
            chunk_map[(fname, idx)] = c

    cited_pairs = set()
    cite_pattern = re.findall(r'《(.+?)——片段\s*(\d+)》', response)
    for fname, idx_str in cite_pattern:
        try:
            idx = int(idx_str)
            cited_pairs.add((fname.strip(), idx))
        except ValueError:
            pass

    fname_only_pattern = re.findall(r'《(.+?)》', response)
    for fname in fname_only_pattern:
        fname = fname.strip()
        if '——' not in fname and '片段' not in fname:
            for (f, idx) in chunk_map.keys():
                if f == fname:
                    cited_pairs.add((f, idx))

    if not cited_pairs:
        logger.info(f"[Filter] LLM 未引用任何来源")
        return [], []

    cited_chunks = []
    for (fname, idx) in cited_pairs:
        key = (fname, idx)
        if key in chunk_map:
            cited_chunks.append(chunk_map[key])

    cited_filenames = {fname for fname, _ in cited_pairs}
    cited_sources = [s for s in sources if s.get("filename", "") in cited_filenames]

    logger.info(f"[Filter] 检索到 {len(chunks)} 个片段，LLM 引用了 {len(cited_chunks)} 个: {cited_pairs}")
    return cited_sources, cited_chunks


def _clean_response(text: str) -> str:
    if not text:
        return text
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and "tool_call" in data:
            return ""
    except (json.JSONDecodeError, ValueError):
        pass
    cleaned = re.sub(r'```json\s*\{"tool_call".*?\}\s*```', '', text, flags=re.DOTALL).strip()
    cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


# ==================== 主Agent（对外接口）====================

class EnterpriseAgent:
    """
    企业智能知识库Agent - 基于 LangGraph 完整自纠错工作流

    工作流：retrieve → generate → evaluate → [pass→END / fail→retry→END]
    流式输出由 chat_stream 从工作流状态中提取。
    """

    def __init__(self, user_id: str = "default", web_search: bool = False):
        self.user_id = user_id
        self.llm = get_mimo_llm_for_agent()
        self.rag_engine = RAGEngine()
        self._web_search_enabled = web_search
        self._lock = threading.Lock()

        self.memory_manager = MemoryManager(
            user_id=user_id,
            llm_fn=self._create_llm_fn()
        )

        # 编译 LangGraph 工作流
        self._workflow = create_agent_workflow(self.llm, self.rag_engine, self.memory_manager)

    def _create_llm_fn(self):
        llm = self.llm
        def llm_fn(messages):
            from langchain_core.messages import HumanMessage, SystemMessage
            lc_messages = [SystemMessage(content="你是一个对话摘要助手，擅长提取关键信息。")]
            for msg in messages:
                lc_messages.append(HumanMessage(content=msg["content"]))
            response = llm.invoke(lc_messages)
            return response.content
        return llm_fn

    def set_web_search(self, enabled: bool):
        with self._lock:
            self._web_search_enabled = enabled

    @property
    def web_search_enabled(self) -> bool:
        return self._web_search_enabled

    def _build_initial_state(self, message: str) -> AgentState:
        return {
            "message": message,
            "user_id": self.user_id,
            "route": "",
            "search_query": "",
            "context": "",
            "sources": [],
            "chunks": [],
            "search_result": "",
            "prompt": "",
            "response": "",
            "eval_result": None,
            "attempt": 0,
            "needs_retry": False,
            "web_search_enabled": self._web_search_enabled,
        }

    def chat_stream(self, message: str, should_stop: Optional[Callable[[], bool]] = None):
        """
        流式对话入口

        运行 LangGraph 完整工作流（检索→生成→评估→重检索→生成），
        从工作流状态中提取结果并流式输出。
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # 运行 LangGraph 工作流（包含完整的 检索→生成→评估→重试 循环）
        initial_state = self._build_initial_state(message)
        accumulated = dict(initial_state)
        node_history = []

        for state_update in self._workflow.stream(initial_state):
            for node_name, partial in state_update.items():
                if isinstance(partial, dict):
                    accumulated.update(partial)
                    node_history.append(node_name)

        route = accumulated.get("route", "internal")
        response = accumulated.get("response", "")
        sources = accumulated.get("sources", [])
        chunks = accumulated.get("chunks", [])
        search_result = accumulated.get("search_result", "")
        eval_result = accumulated.get("eval_result")

        logger.info(f"[Agent] user={self.user_id}, route={route}, nodes={'→'.join(node_history)}, "
                     f"response_len={len(response)}, eval={'yes' if eval_result else 'no'}")

        # 处理 external_disabled
        if route == "external_disabled" or search_result == "__disabled__":
            yield {"type": "sources", "data": {"sources": [], "chunks": []}}
            yield {"type": "chunk", "data": "您询问的内容需要网络搜索，但该功能当前未启用。\n\n请在侧边栏开启「🌐 启用网络搜索」后重试，或直接提问企业内部问题。"}
            return

        # 处理 external 路由的工具调用状态
        if route == "external" and search_result:
            yield {"type": "tool_call", "data": {"tool": "search_web", "args": json.dumps({"query": message}, ensure_ascii=False)}}
            yield {"type": "tool_result", "data": {"tool": "search_web", "result": search_result[:300]}}

        # 流式输出工作流生成的回答
        if response:
            for i in range(0, len(response), 3):
                if should_stop and should_stop():
                    break
                chunk_text = response[i:i+3]
                yield {"type": "chunk", "data": chunk_text}

        # 发送评估结果
        if eval_result:
            yield {"type": "evaluation", "data": eval_result}

            # 降级策略
            if not eval_result.get("overall_passed", True):
                strategy = settings.EVAL_DEGRADATION_STRATEGY
                if strategy == "fallback" and chunks:
                    top1 = chunks[0].get("content", "")[:500]
                    yield {"type": "chunk", "data": f"\n\n📄 降级为 Top-1 原文片段：\n\n{top1}"}

        # 过滤来源
        if sources and response:
            cited_sources, cited_chunks = _filter_cited_sources(response, sources, chunks)
            if cited_sources:
                yield {"type": "sources", "data": {"sources": cited_sources, "chunks": cited_chunks}}
            else:
                yield {"type": "sources", "data": {"sources": sources, "chunks": chunks}}
        elif sources or chunks:
            yield {"type": "sources", "data": {"sources": sources, "chunks": chunks}}

        # 保存记忆
        if response and not (should_stop and should_stop()):
            cleaned = _clean_response(response)
            if cleaned:
                with self._lock:
                    self.memory_manager.add_message("human", message)
                    self.memory_manager.add_message("assistant", cleaned)

    # ==================== 兼容旧接口 ====================

    def add_document(self, file_path: str, metadata: Optional[Dict] = None, content_hash: str = "") -> int:
        return self.rag_engine.add_document(file_path, metadata, content_hash=content_hash)

    def get_knowledge_base_info(self) -> Dict[str, Any]:
        return {"document_count": self.rag_engine.get_document_count(), "memory_stats": self.memory_manager.get_memory_stats()}

    def clear_conversation(self):
        self.memory_manager.clear_conversation_history()
        self.memory_manager.update_conversation_summary("")

    def clear_knowledge_base(self):
        self.rag_engine.clear()
        self.memory_manager.update_conversation_summary("")

    def clear_documents(self):
        self.rag_engine.clear_user_documents(self.user_id)
        self.memory_manager.update_conversation_summary("")
