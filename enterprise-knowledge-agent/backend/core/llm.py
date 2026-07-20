"""LLM 封装模块"""

from backend.config.settings import settings
from functools import lru_cache


@lru_cache()
def get_mimo_llm(
    temperature: float = 0.7,
    streaming: bool = True,
    max_tokens: int = 2048
):
    """
    获取MIMO-v2.5 LLM实例

    Args:
        temperature: 温度参数，控制随机性（0-1）
        streaming: 是否启用流式输出
        max_tokens: 最大生成token数

    Returns:
        ChatOpenAI实例
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.MIMO_MODEL,
        api_key=settings.MIMO_API_KEY,
        base_url=settings.MIMO_BASE_URL,
        temperature=temperature,
        streaming=streaming,
        max_tokens=max_tokens,
    )


def get_mimo_llm_for_agent():
    """获取用于Agent的LLM（中等温度，流式输出，更大 token 上限）"""
    return get_mimo_llm(temperature=0.5, streaming=True, max_tokens=4096)
