"""质量评估模块测试"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.evaluation import QualityEvaluator


@pytest.fixture
def evaluator():
    return QualityEvaluator()


class TestFaithfulness:
    """忠实度评估测试"""

    def test_no_chunks(self, evaluator):
        """无检索结果时通过"""
        result = evaluator.evaluate_faithfulness("回答内容", "上下文", [])
        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_valid_citations(self, evaluator):
        """引用真实来源时通过"""
        answer = "根据《退费指南——片段1》的规定，可以退费。"
        chunks = [{"filename": "退费指南.pdf", "index": 1, "content": "可以退费"}]
        result = evaluator.evaluate_faithfulness(answer, "上下文", chunks)
        assert result["passed"] is True
        assert result["checks"]["valid_cited"] == 1

    def test_citation_with_extension(self, evaluator):
        """LLM引用带扩展名的文件名时也能匹配"""
        answer = "根据《退费指南.pdf——片段1》的规定，可以退费。"
        chunks = [{"filename": "退费指南.pdf", "index": 1, "content": "可以退费"}]
        result = evaluator.evaluate_faithfulness(answer, "上下文", chunks)
        assert result["passed"] is True
        assert result["checks"]["valid_cited"] == 1

    def test_citation_with_timestamp_filename(self, evaluator):
        """LLM引用带时间戳前缀的文件名时也能匹配"""
        answer = "根据《2026年毕业生后勤服务项目退费指南——片段1》的规定。"
        chunks = [{"filename": "20260721_174244_2026年毕业生后勤服务项目退费指南.pdf", "index": 1, "content": "退费规定"}]
        result = evaluator.evaluate_faithfulness(answer, "上下文", chunks)
        assert result["passed"] is True
        assert result["checks"]["valid_cited"] == 1

    def test_invalid_citations(self, evaluator):
        """引用不存在的来源时不通过"""
        answer = "根据《不存在的文件——片段1》的规定。"
        chunks = [{"filename": "退费指南.pdf", "index": 1, "content": "可以退费"}]
        result = evaluator.evaluate_faithfulness(answer, "上下文", chunks)
        assert result["passed"] is False
        assert result["checks"]["valid_cited"] == 0

    def test_partial_citations(self, evaluator):
        """部分引用有效时"""
        answer = "根据《退费指南——片段1》和《不存在——片段2》。"
        chunks = [{"filename": "退费指南.pdf", "index": 1, "content": "可以退费"}]
        result = evaluator.evaluate_faithfulness(answer, "上下文", chunks)
        assert result["checks"]["valid_cited"] == 1
        assert result["checks"]["total_cited"] == 2


class TestRelevancy:
    """答案相关性评估测试"""

    def test_empty_inputs(self, evaluator):
        """空输入时通过"""
        result = evaluator.evaluate_answer_relevancy("", "")
        assert result["passed"] is True

    def test_good_relevancy(self, evaluator):
        """回答包含问题关键术语时通过"""
        question = "寝室电费如何退费"
        answer = "寝室电费退费需要到后勤处办理，携带缴费凭证。"
        result = evaluator.evaluate_answer_relevancy(question, answer)
        assert result["passed"] is True
        assert result["score"] >= 0.5

    def test_poor_relevancy(self, evaluator):
        """回答不包含问题关键术语时不通过"""
        question = "寝室电费如何退费"
        answer = "今天天气很好，适合出门散步。"
        result = evaluator.evaluate_answer_relevancy(question, answer)
        assert result["passed"] is False

    def test_compound_question(self, evaluator):
        """复合问题拆分评估"""
        question = "寝室各种费用如何进行退费，以及我该如何转移共青团组织关系"
        answer = "寝室费用退费需要到后勤处办理。共青团组织关系转移需要开具介绍信。"
        result = evaluator.evaluate_answer_relevancy(question, answer)
        # 应该通过，因为两个子问题都有覆盖
        assert result["score"] >= 0.3  # 至少部分覆盖

    def test_compound_question_partial(self, evaluator):
        """复合问题只回答了一部分"""
        question = "寝室各种费用如何进行退费，以及我该如何转移共青团组织关系"
        answer = "寝室费用退费需要到后勤处办理，携带缴费凭证和身份证。"
        result = evaluator.evaluate_answer_relevancy(question, answer)
        # 第一个子问题覆盖了，第二个没有，平均分应该较低
        assert result["score"] < 0.7

    def test_stop_words_filtered(self, evaluator):
        """停用词被正确过滤"""
        question = "请问如何办理退费手续"
        answer = "办理退费手续需要携带身份证和缴费凭证到财务处。"
        result = evaluator.evaluate_answer_relevancy(question, answer)
        # "请问"、"如何" 应该被过滤
        assert result["passed"] is True


class TestEvaluate:
    """综合评估测试"""

    def test_full_pass(self, evaluator):
        """所有维度都达标"""
        question = "寝室电费如何退费"
        answer = "根据《退费指南——片段1》的规定，寝室电费退费需要到后勤处办理。"
        context = "退费指南规定，寝室电费退费需要到后勤处办理。"
        chunks = [{"filename": "退费指南.pdf", "index": 1, "content": context}]
        result = evaluator.evaluate(question, answer, context, chunks)
        assert result["overall_passed"] is True
        assert result["degradation_strategy"] == "none"

    def test_relevancy_fail(self, evaluator):
        """相关性不达标时触发重试"""
        question = "寝室电费如何退费"
        answer = "今天天气很好。"
        context = "退费指南规定，寝室电费退费需要到后勤处办理。"
        chunks = [{"filename": "退费指南.pdf", "index": 1, "content": context}]
        result = evaluator.evaluate(question, answer, context, chunks)
        assert result["overall_passed"] is False
        assert result["degradation_strategy"] == "retry"

    def test_empty_answer(self, evaluator):
        """空回答时通过（不评估）"""
        result = evaluator.evaluate("问题", "", "上下文", [])
        assert result["overall_passed"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
