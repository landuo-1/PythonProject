"""记忆管理模块 - 管理对话历史、摘要压缩和长期记忆"""

import json
import logging
import os
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

from backend.config.settings import settings

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量（粗略但实用）

    中文约 1.5 字符 = 1 token，英文约 4 字符 = 1 token
    混合文本按 2 字符 = 1 token 估算

    Args:
        text: 输入文本

    Returns:
        估算的 token 数
    """
    if not text:
        return 0
    return max(1, len(text) // 2)


class MemoryManager:
    """
    记忆管理器 - 管理对话历史和长期记忆

    核心能力：
    1. 对话历史管理（追加、检索、滑动窗口）
    2. 自动摘要压缩（旧消息 → 摘要，节省 token）
    3. 长期记忆（用户偏好、知识图谱、对话摘要）
    4. 记忆上下文注入（导出为 system prompt 片段）
    """

    # 滑动窗口配置
    MAX_RECENT_MESSAGES = 10       # 保留最近 N 条消息
    SUMMARY_THRESHOLD = 20         # 消息数超过此阈值时触发摘要压缩
    MAX_HISTORY_MESSAGES = 50      # 历史消息硬上限

    def __init__(self, user_id: str = "default", llm_fn: Optional[Callable] = None):
        """
        初始化记忆管理器

        Args:
            user_id: 用户ID
            llm_fn: 可选的 LLM 调用函数，用于摘要生成。
                     签名: llm_fn(messages: List[Dict]) -> str
        """
        import re
        import threading
        # 用户ID安全处理（防止路径遍历）
        self.user_id = re.sub(r'[^\w\-]', '_', user_id)
        self.llm_fn = llm_fn
        self._save_lock = threading.Lock()
        self._compress_lock = threading.Lock()  # 保护摘要压缩（防止并发触发）

        # 统一使用 backend/data/memory 目录
        self.memory_dir = Path(settings.UPLOAD_DIR).parent / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # 对话历史（内存）
        self.conversation_history: List[Dict[str, Any]] = []

        # 长期记忆（持久化到 JSON）
        self.long_term_memory: Dict[str, Any] = self._load_long_term_memory()

    # ==================== 持久化 ====================

    def _get_memory_file_path(self) -> Path:
        """获取记忆文件路径"""
        return self.memory_dir / f"{self.user_id}_memory.json"

    def _load_long_term_memory(self) -> Dict[str, Any]:
        """加载长期记忆"""
        file_path = self._get_memory_file_path()

        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载记忆失败: {e}")

        return {
            "user_preferences": {},
            "knowledge_graph": {},
            "conversation_summary": "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

    def _save_long_term_memory(self):
        """保存长期记忆（带并发锁）"""
        file_path = self._get_memory_file_path()

        with self._save_lock:
            try:
                self.long_term_memory["updated_at"] = datetime.now().isoformat()
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"保存记忆失败: {e}")

    # ==================== 对话历史管理 ====================

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        添加消息到对话历史

        Args:
            role: 角色 (human/assistant/system)
            content: 消息内容
            metadata: 额外元数据
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.conversation_history.append(message)

        # 限制历史长度
        if len(self.conversation_history) > self.MAX_HISTORY_MESSAGES:
            self.conversation_history = self.conversation_history[-self.MAX_HISTORY_MESSAGES:]

    def get_conversation_history(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """
        获取对话历史

        Args:
            max_messages: 最大消息数

        Returns:
            对话历史列表
        """
        return self.conversation_history[-max_messages:]

    def get_langchain_messages(self, max_messages: int = 10) -> List:
        """
        获取LangChain格式的消息

        Args:
            max_messages: 最大消息数

        Returns:
            LangChain消息列表
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        messages = []
        history = self.get_conversation_history(max_messages)

        for msg in history:
            role = msg["role"]
            content = msg["content"]

            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))

        return messages

    # ==================== 摘要压缩 ====================

    def summarize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """
        将一组消息压缩成摘要

        Args:
            messages: 待压缩的消息列表

        Returns:
            摘要字符串
        """
        if not messages:
            return ""

        # 如果没有 LLM，用简单的截断方式
        if self.llm_fn is None:
            return self._simple_summarize(messages)

        # 用 LLM 生成摘要
        try:
            summary_prompt = self._build_summary_prompt(messages)
            summary = self.llm_fn([{"role": "user", "content": summary_prompt}])
            return summary
        except Exception as e:
            logger.warning(f"LLM摘要生成失败，使用简单摘要: {e}")
            return self._simple_summarize(messages)

    def _simple_summarize(self, messages: List[Dict[str, Any]]) -> str:
        """
        简单摘要（不依赖 LLM）

        Args:
            messages: 消息列表

        Returns:
            简单摘要
        """
        topics = []
        for msg in messages:
            if msg["role"] == "human":
                # 取用户消息的前50字符作为话题
                content = msg["content"][:50]
                if len(msg["content"]) > 50:
                    content += "..."
                topics.append(content)

        if not topics:
            return ""

        return f"之前的对话涉及以下话题：{'；'.join(topics)}"

    def _build_summary_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        构建摘要生成的 prompt

        Args:
            messages: 消息列表

        Returns:
            prompt 字符串
        """
        conversation_text = ""
        for msg in messages:
            role = "用户" if msg["role"] == "human" else "助手"
            conversation_text += f"{role}: {msg['content']}\n"

        return f"""请将以下对话压缩成一段简洁的摘要，保留关键信息（用户问了什么、得出了什么结论、用户的偏好）。

对话内容：
{conversation_text}

要求：
- 用中文输出
- 保留关键事实和结论
- 如果用户表达了偏好，记录下来
- 长度不超过200字"""

    # ==================== 滑动窗口 ====================

    def get_context_messages(self) -> List[Dict[str, Any]]:
        """
        获取上下文消息（滑动窗口策略）

        策略：
        - 消息数 < SUMMARY_THRESHOLD: 返回全部历史
        - 消息数 >= SUMMARY_THRESHOLD:
          1. 取前 N-MAX_RECENT_MESSAGES 条旧消息 → 生成摘要
          2. 截断 conversation_history，保留最近消息 + 摘要
          3. 返回 [摘要] + 最近消息

        Returns:
            上下文消息列表（用于传给 LLM）
        """
        if len(self.conversation_history) < self.SUMMARY_THRESHOLD:
            return self.conversation_history

        # 需要压缩（加锁防止并发触发多次压缩）
        with self._compress_lock:
            # 双重检查：获取锁后再次检查阈值（可能其他线程已完成压缩）
            if len(self.conversation_history) < self.SUMMARY_THRESHOLD:
                return self.conversation_history

            split_point = len(self.conversation_history) - self.MAX_RECENT_MESSAGES
            old_messages = self.conversation_history[:split_point]
            recent_messages = self.conversation_history[split_point:]

            logger.info(f"[MEMORY] 触发对话压缩: 总消息={len(self.conversation_history)}, 压缩={len(old_messages)}条, 保留={len(recent_messages)}条")

            # 生成摘要
            summary = self.summarize_messages(old_messages)

            # 压缩成功后截断历史，避免下次重复压缩同一批旧消息
            if summary:
                summary_msg = {
                    "role": "system",
                    "content": f"[对话历史摘要] {summary}",
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {"type": "summary"}
                }
                self.conversation_history = [summary_msg] + recent_messages
                self.update_conversation_summary(summary)
                logger.info(f"[MEMORY] 压缩完成: 摘要长度={len(summary)}, 历史截断至{len(self.conversation_history)}条")
            else:
                logger.warning("[MEMORY] 摘要生成为空，保留原始历史")

        # 返回当前历史（已包含摘要）
        return self.conversation_history

    def get_token_count(self) -> int:
        """
        估算当前上下文的 token 总数

        Returns:
            估算的 token 数
        """
        total = 0
        for msg in self.conversation_history:
            total += estimate_tokens(msg.get("content", ""))
        return total

    # ==================== 长期记忆 ====================

    def update_user_preference(self, key: str, value: Any):
        """更新用户偏好"""
        self.long_term_memory["user_preferences"][key] = value
        self._save_long_term_memory()

    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """获取用户偏好"""
        return self.long_term_memory["user_preferences"].get(key, default)

    def add_knowledge(self, entity: str, relation: str, value: str):
        """添加知识到知识图谱"""
        if entity not in self.long_term_memory["knowledge_graph"]:
            self.long_term_memory["knowledge_graph"][entity] = []

        self.long_term_memory["knowledge_graph"][entity].append({
            "relation": relation,
            "value": value,
            "added_at": datetime.now().isoformat()
        })

        self._save_long_term_memory()

    def get_knowledge(self, entity: str) -> List[Dict]:
        """获取实体相关知识"""
        return self.long_term_memory["knowledge_graph"].get(entity, [])

    def update_conversation_summary(self, summary: str):
        """更新对话摘要"""
        self.long_term_memory["conversation_summary"] = summary
        self._save_long_term_memory()

    def get_conversation_summary(self) -> str:
        """获取对话摘要"""
        return self.long_term_memory.get("conversation_summary", "")

    # ==================== 记忆上下文注入 ====================

    def get_memory_context_for_prompt(self) -> str:
        """
        导出记忆上下文，用于注入 system prompt

        Returns:
            格式化的记忆上下文字符串
        """
        parts = []

        # 1. 用户偏好
        preferences = self.long_term_memory.get("user_preferences", {})
        if preferences:
            pref_text = "、".join([f"{k}:{v}" for k, v in preferences.items()])
            parts.append(f"用户偏好：{pref_text}")

        # 2. 对话摘要
        summary = self.get_conversation_summary()
        if summary:
            parts.append(f"历史对话摘要：{summary}")

        # 3. 知识图谱
        knowledge = self.long_term_memory.get("knowledge_graph", {})
        if knowledge:
            knowledge_items = []
            for entity, relations in knowledge.items():
                for rel in relations[:3]:  # 每个实体最多3条
                    knowledge_items.append(f"{entity} → {rel['relation']} → {rel['value']}")
            if knowledge_items:
                parts.append(f"已知信息：{'；'.join(knowledge_items[:5])}")

        # 4. 对话统计
        msg_count = len(self.conversation_history)
        if msg_count > 0:
            parts.append(f"当前对话已进行 {msg_count} 轮")

        if not parts:
            return ""

        return "\n".join(parts)

    # ==================== 工具方法 ====================

    def clear_conversation_history(self):
        """清空对话历史"""
        self.conversation_history = []

    def clear_all_memory(self):
        """清空所有记忆"""
        self.conversation_history = []
        self.long_term_memory = {
            "user_preferences": {},
            "knowledge_graph": {},
            "conversation_summary": "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self._save_long_term_memory()

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "conversation_count": len(self.conversation_history),
            "preference_count": len(self.long_term_memory["user_preferences"]),
            "knowledge_count": sum(
                len(v) for v in self.long_term_memory["knowledge_graph"].values()
            ),
            "has_summary": bool(self.long_term_memory.get("conversation_summary")),
            "estimated_tokens": self.get_token_count(),
            "last_updated": self.long_term_memory.get("updated_at")
        }

    def export_memory(self) -> Dict[str, Any]:
        """导出记忆数据"""
        return {
            "user_id": self.user_id,
            "conversation_history": self.conversation_history,
            "long_term_memory": self.long_term_memory,
            "exported_at": datetime.now().isoformat()
        }

    def import_memory(self, data: Dict[str, Any]):
        """导入记忆数据"""
        if "conversation_history" in data:
            self.conversation_history = data["conversation_history"]
        if "long_term_memory" in data:
            self.long_term_memory = data["long_term_memory"]
            self._save_long_term_memory()
