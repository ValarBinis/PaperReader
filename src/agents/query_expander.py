"""
QueryExpander - AI智能搜索词拓展模块
使用LLM分析研究主题，生成相关搜索关键词
"""

import json
import re
from typing import List

from ..apis.llm_client import LLMClient
from ..utils.config import get_prompts


class QueryExpander:
    """
    AI智能搜索词拓展器
    分析研究主题，生成多个相关搜索关键词
    """

    def __init__(self, llm_client: LLMClient):
        """
        初始化QueryExpander

        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
        self.prompts = get_prompts()

    def expand_query(self, topic: str, num_keywords: int = 10) -> List[str]:
        """
        拓展搜索关键词

        Args:
            topic: 研究主题
            num_keywords: 生成关键词数量

        Returns:
            搜索关键词列表
        """
        # 获取prompt模板
        prompt_template = self.prompts.get("query_expansion_prompt",
            topic=topic, num_keywords=num_keywords)

        # 调用LLM
        messages = [
            {
                "role": "system",
                "content": "你是学术搜索专家，擅长分析研究主题并生成相关搜索关键词。"
            },
            {
                "role": "user",
                "content": prompt_template
            }
        ]

        try:
            response = self.llm_client.chat(messages)

            # 解析JSON响应
            keywords = self._parse_keywords(response)

            # 确保包含原始主题
            if topic.lower() not in [k.lower() for k in keywords]:
                keywords.insert(0, topic)

            return keywords[:num_keywords]

        except Exception as e:
            print(f"搜索词拓展失败: {e}")
            return [topic]

    def expand_query_with_feedback(
        self,
        topic: str,
        failed_keywords: List[str],
        num_keywords: int = 10
    ) -> List[str]:
        """
        根据之前的失败关键词，重新生成搜索词

        Args:
            topic: 研究主题
            failed_keywords: 之前使用但失败的关键词列表
            num_keywords: 生成关键词数量

        Returns:
            搜索关键词列表
        """
        # 构建带反馈的prompt
        failed_keywords_str = "\n".join([f"- {kw}" for kw in failed_keywords[:20]])  # 限制显示数量

        prompt = f"""你是学术搜索专家。我正在arXiv上搜索关于"{topic}"的论文。

以下关键词搜索不到任何结果：
{failed_keywords_str}

请分析为什么这些关键词搜不到论文，并生成{num_keywords}个新的、更有效的搜索关键词。

注意事项：
1. arXiv主要收录计算机科学、物理、数学、定量生物学、定量金融、统计学、电气工程、经济学等领域的论文
2. 如果主题是金融/经济类，请使用英文专业术语，如: "stock valuation", "asset pricing", "financial modeling", "portfolio optimization" 等
3. 搜索词应该简洁、专业，避免使用中文
4. 可以尝试相关领域的交叉概念

请直接返回JSON数组格式，例如：
["keyword1", "keyword2", "keyword3", ...]

不要包含任何其他解释文字。
"""

        messages = [
            {
                "role": "system",
                "content": "你是学术搜索专家，擅长分析研究主题并生成相关搜索关键词。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = self.llm_client.chat(messages)

            # 解析JSON响应
            keywords = self._parse_keywords(response)

            # 确保包含原始主题的英文翻译
            if topic.lower() not in [k.lower() for k in keywords]:
                keywords.insert(0, topic)

            return keywords[:num_keywords]

        except Exception as e:
            print(f"搜索词拓展失败: {e}")
            # 回退到英文翻译
            return [topic, "stock valuation", "asset pricing", "financial analysis"]

    def _parse_keywords(self, response: str) -> List[str]:
        """
        解析LLM返回的关键词

        Args:
            response: LLM响应文本

        Returns:
            关键词列表
        """
        # 尝试解析JSON
        json_match = re.search(r'\[([^\]]+)\]', response)
        if json_match:
            try:
                keywords = json.loads(f"[{json_match.group(1)}]")
                return [str(k).strip('"\'') for k in keywords]
            except json.JSONDecodeError:
                pass

        # 尝试按行解析
        lines = response.split('\n')
        keywords = []
        for line in lines:
            line = line.strip()
            # 移除编号和符号
            line = re.sub(r'^[\d\-\*\•]+\.?\s*', '', line)
            line = re.sub(r'^["\'\[\]]+|["\'\[\]]+$', '', line)
            if line and len(line) > 2:
                keywords.append(line)

        return keywords

    def expand_with_categories(
        self,
        topic: str,
        categories: List[str] = None
    ) -> dict:
        """
        按分类拓展搜索词

        Args:
            topic: 研究主题
            categories: arXiv分类列表

        Returns:
            包含分类和关键词的字典
        """
        prompt = f"""分析研究主题"{topic}"，生成搜索关键词。

请按以下分类返回JSON格式：
{{
    "core_concepts": ["核心概念1", "核心概念2"],
    "techniques": ["技术方法1", "技术方法2"],
    "applications": ["应用场景1", "应用场景2"],
    "authors": ["重要作者1", "重要作者2"],
    "papers": ["经典论文1", "经典论文2"]
}}

每个分类生成2-4个关键词。
"""

        messages = [
            {
                "role": "system",
                "content": "你是学术搜索专家。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = self.llm_client.chat(messages)

            # 解析JSON
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))

                # 添加arXiv分类
                if categories:
                    result["arxiv_categories"] = categories

                return result

        except Exception as e:
            print(f"分类拓展失败: {e}")

        # 失败时返回简单格式
        keywords = self.expand_query(topic)
        return {
            "core_concepts": keywords[:3],
            "techniques": keywords[3:6],
            "applications": keywords[6:8],
            "authors": [],
            "papers": [],
            "arxiv_categories": categories or []
        }

    def generate_arxiv_queries(
        self,
        topic: str,
        categories: List[str] = None
    ) -> List[str]:
        """
        生成arXiv搜索查询

        Args:
            topic: 研究主题
            categories: arXiv分类

        Returns:
            arXiv搜索查询列表
        """
        expanded = self.expand_with_categories(topic, categories)

        queries = []

        # 核心概念查询
        for concept in expanded.get("core_concepts", []):
            queries.append(f"all:{concept}")

        # 技术方法查询
        for technique in expanded.get("techniques", []):
            queries.append(f"all:{technique}")

        # 按分类查询
        if categories:
            for cat in categories:
                queries.append(f"cat:{cat}")

        # 经典论文查询
        for paper in expanded.get("papers", []):
            queries.append(f"all:{paper}")

        return list(set(queries))  # 去重


def expand_query(topic: str, llm_client: LLMClient) -> List[str]:
    """
    便捷函数：拓展搜索关键词

    Args:
        topic: 研究主题
        llm_client: LLM客户端

    Returns:
        关键词列表
    """
    expander = QueryExpander(llm_client)
    return expander.expand_query(topic)
