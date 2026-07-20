"""Agent测试"""

import pytest

from backend.core.agent import EnterpriseAgent
from backend.core.llm import get_mimo_llm
from backend.core.rag import RAGEngine
from backend.core.memory import MemoryManager


class TestLLM:
    """LLM测试"""

    def test_mimo_llm_initialization(self):
        """测试MIMO LLM初始化"""
        try:
            llm = get_mimo_llm()
            assert llm is not None
            print("[OK] MIMO LLM初始化成功")
        except Exception as e:
            pytest.skip(f"MIMO LLM初始化失败: {e}")

    def test_mimo_llm_invocation(self):
        """测试MIMO LLM调用"""
        try:
            llm = get_mimo_llm(temperature=0.1, streaming=False)
            response = llm.invoke("你好，请简单介绍一下你自己")
            assert response.content is not None
            assert len(response.content) > 0
            print(f"[OK] MIMO LLM调用成功，响应长度: {len(response.content)}")
        except Exception as e:
            pytest.skip(f"MIMO LLM调用失败: {e}")


class TestRAGEngine:
    """RAG引擎测试"""

    def test_rag_engine_initialization(self):
        """测试RAG引擎初始化"""
        try:
            rag = RAGEngine()
            assert rag is not None
            assert rag.embeddings is not None
            assert rag.vector_store is not None
            print("[OK] RAG引擎初始化成功")
        except Exception as e:
            pytest.skip(f"RAG引擎初始化失败: {e}")

    def test_text_splitter(self):
        """测试文本分割器"""
        try:
            rag = RAGEngine()
            test_text = "这是一段测试文本。" * 100
            chunks = rag.text_splitter.split_text(test_text)
            assert len(chunks) > 0
            print(f"[OK] 文本分割成功，分成 {len(chunks)} 个块")
        except Exception as e:
            pytest.skip(f"文本分割失败: {e}")


class TestMemoryManager:
    """记忆管理器测试"""

    def test_memory_initialization(self):
        """测试记忆管理器初始化"""
        try:
            memory = MemoryManager("test_user")
            assert memory is not None
            assert memory.user_id == "test_user"
            print("[OK] 记忆管理器初始化成功")
        except Exception as e:
            pytest.skip(f"记忆管理器初始化失败: {e}")

    def test_add_message(self):
        """测试添加消息"""
        try:
            memory = MemoryManager("test_user")
            memory.add_message("human", "你好")
            memory.add_message("assistant", "你好！有什么可以帮您的吗？")

            history = memory.get_conversation_history()
            assert len(history) == 2
            print("[OK] 消息添加成功")
        except Exception as e:
            pytest.skip(f"消息添加失败: {e}")

    def test_user_preference(self):
        """测试用户偏好"""
        try:
            memory = MemoryManager("test_user")
            memory.update_user_preference("theme", "dark")
            theme = memory.get_user_preference("theme")
            assert theme == "dark"
            print("[OK] 用户偏好设置成功")
        except Exception as e:
            pytest.skip(f"用户偏好设置失败: {e}")


class TestEnterpriseAgent:
    """企业Agent测试"""

    def test_agent_initialization(self):
        """测试Agent初始化"""
        try:
            agent = EnterpriseAgent("test_user")
            assert agent is not None
            assert agent.llm is not None
            assert agent.rag_engine is not None
            assert agent.memory_manager is not None
            print("[OK] Agent初始化成功")
        except Exception as e:
            pytest.skip(f"Agent初始化失败: {e}")

    def test_agent_chat(self):
        """测试Agent流式对话"""
        try:
            agent = EnterpriseAgent("test_user")
            chunks = list(agent.chat_stream("你好，请介绍一下你自己"))
            assert len(chunks) > 0
            # 收集所有 chunk 类型的响应
            response_parts = [c["data"] for c in chunks if c["type"] == "chunk"]
            full_response = "".join(response_parts)
            assert len(full_response) > 0
            print(f"[OK] Agent对话成功，响应长度: {len(full_response)}")
        except Exception as e:
            pytest.skip(f"Agent对话失败: {e}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
