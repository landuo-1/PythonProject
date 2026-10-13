"""工具模块 - 网络搜索功能"""

import logging
import os
import re
from backend.config.settings import settings

logger = logging.getLogger(__name__)

# 设置 Tavily 环境变量
if settings.TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY


def search_web(query: str) -> str:
    """
    搜索互联网获取最新信息

    自动根据查询内容判断时效性需求：
    - 强时效性关键词（今天/最新/实时）→ time_range="day"
    - 一般时效性关键词（本周/近期）→ time_range="week"
    - 通用查询 → 不限时间

    Args:
        query: 搜索查询

    Returns:
        搜索结果
    """
    try:
        from langchain_tavily import TavilySearch
        import concurrent.futures

        if not os.environ.get("TAVILY_API_KEY"):
            logger.warning("[TOOL] Tavily API Key 未配置")
            return "搜索功能不可用：未配置 Tavily API Key"

        # 根据查询关键词判断时效性需求
        time_range = None
        if re.search(r'(今天|今日|刚刚|实时|现在|当前|latest|today|now)', query, re.IGNORECASE):
            time_range = "day"
        elif re.search(r'(本周|这周|近期|最近|recent|this week)', query, re.IGNORECASE):
            time_range = "week"

        # 构建搜索参数
        search_kwargs = {"max_results": 3, "search_depth": "advanced"}
        if time_range:
            search_kwargs["time_range"] = time_range
            logger.info(f"[TOOL] 搜索时效性: time_range={time_range}")

        search = TavilySearch(**search_kwargs)
        # 超时保护
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(search.invoke, query)
            try:
                result = future.result(timeout=15)
            except concurrent.futures.TimeoutError:
                logger.warning("[TOOL] Tavily 搜索超时（15s）")
                return "搜索超时，请稍后重试"

        # 解析结果
        if isinstance(result, dict):
            results = result.get("results", [])
        elif isinstance(result, list):
            results = result
        else:
            return "搜索返回了未知格式的结果"

        if not results:
            return "未找到相关搜索结果"

        formatted_results = ["⚠️ 以下为外部网络信息，仅供参考：\n"]
        for i, item in enumerate(results, 1):
            title = item.get("title", "无标题")
            url = item.get("url", "")
            content = item.get("content", "")
            published = item.get("published_date", "")
            date_info = f"   发布时间: {published}\n" if published else ""
            formatted_results.append(f"{i}. {title}\n   链接: {url}\n{date_info}   摘要: {content}")

        return "\n\n".join(formatted_results)

    except ImportError:
        logger.error("[TOOL] langchain-tavily 未安装")
        return "搜索功能不可用：请安装 langchain-tavily"
    except Exception as e:
        logger.error(f"[TOOL] 搜索异常: {e}")
        return f"搜索失败: {str(e)}"
