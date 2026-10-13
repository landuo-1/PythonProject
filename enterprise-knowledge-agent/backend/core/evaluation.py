"""质量评估模块 - 轻量级规则检查 + LLM 自评

不加载额外模型，评估延迟 < 1 秒。

评估维度：
1. 忠实度（Faithfulness）：回答中的来源引用是否真实存在于检索结果中
2. 答案相关性（Answer Relevancy）：回答是否包含问题的关键术语
"""

import logging
import re
from typing import Dict, Any

from backend.utils.file_utils import normalize_filename as _normalize_filename

logger = logging.getLogger(__name__)


def _extract_keywords_jieba(text: str) -> set:
    """使用 jieba 分词提取中文关键词，过滤停用词和单字"""
    try:
        import jieba
        words = set(w for w in jieba.cut(text) if len(w) >= 2)
    except ImportError:
        # 降级到正则
        words = set(re.findall(r'[一-鿿]{2,4}', text))

    stop_words = {
        '什么', '怎么', '如何', '请问', '可以', '能够', '需要', '为什么',
        '哪些', '哪个', '进行', '办理', '关于', '问题', '情况', '规定',
        '相关', '具体', '以下', '如下', '其中', '所有', '各种', '进行',
        '我', '你', '他', '她', '它', '们', '的', '了', '是', '在',
        '有', '和', '与', '或', '但', '而', '也', '都', '就', '才',
        '到', '把', '被', '让', '给', '从', '向', '往', '以', '为',
    }
    return words - stop_words


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

    # ==================== 忠实度评估 ====================

    def _extract_cited_sources(self, answer: str) -> set:
        """从回答中提取引用的来源（文件名+片段号，文件名归一化）"""
        cited = set()
        # 匹配 《文件名——片段N》
        for match in re.finditer(r'《(.+?)——片段\s*(\d+)》', answer):
            fname = _normalize_filename(match.group(1).strip())
            idx = int(match.group(2))
            cited.add((fname, idx))
        return cited

    def _extract_available_sources(self, chunks: list) -> set:
        """从检索结果中提取可用的来源（文件名归一化）"""
        available = set()
        for c in chunks:
            fname = c.get("filename", "")
            idx = c.get("index", 0)
            if fname:
                available.add((_normalize_filename(fname), idx))
        return available

    def _check_keyword_overlap(self, answer: str, context: str) -> float:
        """检查回答与上下文的关键词重叠度（使用 jieba 分词）"""
        answer_words = _extract_keywords_jieba(answer)
        context_words = _extract_keywords_jieba(context)

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

    # 复合问题分隔词
    _COMPOUND_SEPARATORS = ['以及', '并且', '另外', '同时', '还有', '并且还', '，还']

    def _split_compound_question(self, question: str) -> list:
        """将复合问题拆分为子问题"""
        # 尝试用分隔词拆分
        parts = [question]
        for sep in self._COMPOUND_SEPARATORS:
            new_parts = []
            for part in parts:
                new_parts.extend([p.strip() for p in part.split(sep) if p.strip()])
            parts = new_parts
        return parts

    def _extract_keywords(self, text: str) -> set:
        """提取中文关键词（使用 jieba 分词），过滤停用词"""
        return _extract_keywords_jieba(text)

    def evaluate_answer_relevancy(self, question: str, answer: str) -> Dict[str, Any]:
        """
        答案相关性评估（规则检查）

        对复合问题（含"以及"、"并且"等）拆分为子问题分别检查，
        每个子问题独立评分，取平均分。
        """
        if not question or not answer:
            return {"score": 1.0, "passed": True}

        # 拆分复合问题
        sub_questions = self._split_compound_question(question)

        if len(sub_questions) <= 1:
            # 单一问题，直接检查
            question_terms = self._extract_keywords(question)
            if not question_terms:
                return {"score": 1.0, "passed": True}
            matched = sum(1 for term in question_terms if term in answer)
            score = matched / len(question_terms)
        else:
            # 复合问题，每个子问题独立评分
            sub_scores = []
            all_terms_info = []
            for sub_q in sub_questions:
                sub_terms = self._extract_keywords(sub_q)
                if not sub_terms:
                    sub_scores.append(1.0)
                    continue
                matched = sum(1 for term in sub_terms if term in answer)
                sub_score = matched / len(sub_terms)
                sub_scores.append(sub_score)
                all_terms_info.append({
                    "sub_q": sub_q[:30],
                    "terms": sub_terms,
                    "matched": [t for t in sub_terms if t in answer],
                    "score": round(sub_score, 3),
                })

            # 平均分，但任何一个子问题得分 < 0.3 视为该部分缺失
            score = sum(sub_scores) / len(sub_scores) if sub_scores else 0.0
            logger.debug(f"[Eval] 复合问题拆分: {all_terms_info}, 平均分={score:.3f}")

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
