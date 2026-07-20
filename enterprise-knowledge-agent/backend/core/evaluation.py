"""质量评估模块 - 轻量级规则检查 + LLM 自评

不加载额外模型，评估延迟 < 1 秒。

评估维度：
1. 忠实度（Faithfulness）：回答中的来源引用是否真实存在于检索结果中
2. 答案相关性（Answer Relevancy）：回答是否包含问题的关键术语
"""

import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)


class QualityEvaluator:
    """
    轻量级质量评估器

    使用规则检查 + 可选 LLM 自评，不加载额外模型。
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "QualityEvaluator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def preload(self):
        """兼容接口（轻量评估无需预加载）"""
        logger.info("[Eval] 轻量评估器，无需预加载模型")

    # ==================== 忠实度评估 ====================

    def _extract_cited_sources(self, answer: str) -> set:
        """从回答中提取引用的来源（文件名+片段号）"""
        cited = set()
        # 匹配 《文件名——片段N》
        for match in re.finditer(r'《(.+?)——片段\s*(\d+)》', answer):
            cited.add((match.group(1).strip(), int(match.group(2))))
        return cited

    def _extract_available_sources(self, chunks: list) -> set:
        """从检索结果中提取可用的来源"""
        available = set()
        for c in chunks:
            fname = c.get("filename", "")
            idx = c.get("index", 0)
            if fname:
                available.add((fname, idx))
        return available

    def _check_keyword_overlap(self, answer: str, context: str) -> float:
        """检查回答与上下文的关键词重叠度"""
        # 提取中文关键词（2-4字）
        answer_words = set(re.findall(r'[一-鿿]{2,4}', answer))
        context_words = set(re.findall(r'[一-鿿]{2,4}', context))

        if not answer_words:
            return 0.0

        overlap = answer_words & context_words
        return len(overlap) / len(answer_words)

    def evaluate_faithfulness(self, answer: str, context: str, chunks: list) -> Dict[str, Any]:
        """
        忠实度评估（规则检查）

        检查：
        1. 回答中引用的来源是否真实存在于检索结果中
        2. 回答内容与上下文的关键词重叠度
        """
        if not answer or not chunks:
            return {"score": 1.0, "checks": {}, "passed": True}

        # 检查1：来源引用准确性
        cited = self._extract_cited_sources(answer)
        available = self._extract_available_sources(chunks)

        if cited:
            # 有引用时，检查引用是否都存在于检索结果中
            valid_citations = cited & available
            citation_accuracy = len(valid_citations) / len(cited) if cited else 1.0
        else:
            # 无引用时，用关键词重叠度评估
            keyword_overlap = self._check_keyword_overlap(answer, context)
            citation_accuracy = keyword_overlap

        # 检查2：回答长度合理性（太短可能没回答，太长可能编造）
        length_score = 1.0
        if len(answer) < 20:
            length_score = 0.5
        elif len(answer) > 2000:
            length_score = 0.8

        # 综合分数
        score = citation_accuracy * 0.8 + length_score * 0.2

        return {
            "score": round(score, 3),
            "checks": {
                "citation_accuracy": round(citation_accuracy, 3),
                "total_cited": len(cited),
                "valid_cited": len(cited & available) if cited else 0,
                "keyword_overlap": round(self._check_keyword_overlap(answer, context), 3),
            },
            "passed": score >= 0.70,
        }

    # ==================== 答案相关性评估 ====================

    def evaluate_answer_relevancy(self, question: str, answer: str) -> Dict[str, Any]:
        """
        答案相关性评估（规则检查）

        检查回答是否包含问题中的关键术语。
        """
        if not question or not answer:
            return {"score": 1.0, "passed": True}

        # 提取问题中的关键术语
        question_terms = set(re.findall(r'[一-鿿]{2,6}', question))
        # 过滤停用词
        stop_words = {'什么', '怎么', '如何', '请问', '可以', '能够', '需要', '为什么', '哪些', '哪个'}
        question_terms -= stop_words

        if not question_terms:
            return {"score": 1.0, "passed": True}

        # 检查回答中是否包含问题术语
        matched = sum(1 for term in question_terms if term in answer)
        score = matched / len(question_terms)

        return {
            "score": round(score, 3),
            "passed": score >= 0.50,
        }

    # ==================== 综合评估 ====================

    def evaluate(self, question: str, answer: str, context: str,
                 chunks: list = None, llm_fn=None,
                 faithfulness_threshold: float = 0.70,
                 relevancy_threshold: float = 0.50) -> Dict[str, Any]:
        """
        综合质量评估

        Args:
            question: 用户原始问题
            answer: Agent 回答
            context: 检索到的上下文
            chunks: 检索到的片段列表（用于来源验证）
            llm_fn: 未使用（兼容接口）
            faithfulness_threshold: 忠实度阈值
            relevancy_threshold: 答案相关性阈值
        """
        if not answer or not context:
            return {
                "faithfulness": {"score": 1.0, "checks": {}, "passed": True},
                "relevancy": {"score": 1.0, "passed": True},
                "overall_passed": True,
                "degradation_strategy": "none",
            }

        faithfulness = self.evaluate_faithfulness(answer, context, chunks or [])
        relevancy = self.evaluate_answer_relevancy(question, answer)

        overall_passed = faithfulness["passed"] and relevancy["passed"]

        degradation_strategy = "none"
        if not overall_passed:
            if not faithfulness["passed"]:
                logger.warning(f"[Eval] 忠实度不达标: {faithfulness['score']} < {faithfulness_threshold}")
            if not relevancy["passed"]:
                logger.warning(f"[Eval] 答案相关性不达标: {relevancy['score']} < {relevancy_threshold}")
            degradation_strategy = "retry"

        logger.info(f"[Eval] 忠实度={faithfulness['score']}, 相关性={relevancy['score']}, "
                     f"达标={overall_passed}, 策略={degradation_strategy}")

        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "overall_passed": bool(overall_passed),
            "degradation_strategy": str(degradation_strategy),
        }
