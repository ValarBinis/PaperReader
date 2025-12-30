"""
论文分析器模块
使用LLM分析论文内容
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
import hashlib

from ..apis.llm_client import LLMClient
from ..apis.arxiv_api import ArxivPaper
from ..apis.paper_base import BasePaper
from ..parsers.pdf_parser import PDFParser
from ..utils.config import get_prompts


@dataclass
class PaperAnalysis:
    """论文分析结果数据类"""

    # 基本信息
    title: str = ""
    authors: str = ""
    published: str = ""
    arxiv_id: str = ""
    arxiv_url: str = ""

    # 新增：通用论文ID（支持非arXiv论文）
    paper_id: str = ""
    paper_url: str = ""
    doi: str = ""
    source: str = ""  # 论文来源（arxiv, scihub等）

    # 分析结果
    category: str = ""
    core_problem: str = ""
    contributions: List[str] = field(default_factory=list)
    method_overview: str = ""
    experimental_results: str = ""
    limitations: str = ""

    # 摘要
    summary: str = ""
    full_analysis: str = ""

    # 参考文献目录（用于后续挖掘）
    references: List[str] = field(default_factory=list)

    # 元数据
    analyzed_at: str = ""
    analysis_type: str = "abstract"  # abstract 或 full

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "title": self.title,
            "authors": self.authors,
            "published": self.published,
            "arxiv_id": self.arxiv_id,
            "arxiv_url": self.arxiv_url,
            "paper_id": self.paper_id,
            "paper_url": self.paper_url,
            "doi": self.doi,
            "source": self.source,
            "category": self.category,
            "core_problem": self.core_problem,
            "contributions": self.contributions,
            "method_overview": self.method_overview,
            "experimental_results": self.experimental_results,
            "limitations": self.limitations,
            "summary": self.summary,
            "full_analysis": self.full_analysis,
            "references": self.references,
            "analyzed_at": self.analyzed_at,
            "analysis_type": self.analysis_type
        }


class PaperAnalyzer:
    """
    论文分析器
    使用LLM分析论文内容
    """

    def __init__(
        self,
        llm_client: LLMClient,
        analyze_full_text: bool = True,
        max_pages: int = 0
    ):
        """
        初始化论文分析器

        Args:
            llm_client: LLM客户端
            analyze_full_text: 是否分析全文
            max_pages: 最大分析页数（0表示不限）
        """
        self.llm_client = llm_client
        self.analyze_full_text = analyze_full_text
        self.max_pages = max_pages
        self.prompts = get_prompts()

    def analyze_from_arxiv(
        self,
        paper: ArxivPaper,
        pdf_path: Optional[str] = None
    ) -> PaperAnalysis:
        """
        分析arXiv论文

        Args:
            paper: ArxivPaper对象
            pdf_path: PDF文件路径（如果已下载）

        Returns:
            PaperAnalysis对象
        """
        analysis = PaperAnalysis(
            title=paper.title,
            authors=paper.authors_str,
            published=paper.published.strftime("%Y-%m-%d") if paper.published else "",
            arxiv_id=paper.arxiv_id,
            arxiv_url=paper.arxiv_url,
            paper_id=paper.paper_id,
            paper_url=paper.paper_url,
            source="arxiv"
        )

        # 如果有PDF文件，分析全文
        if pdf_path and Path(pdf_path).exists():
            return self.analyze_from_pdf(pdf_path, analysis)

        # 否则只分析摘要
        return self.analyze_abstract(paper.summary, analysis)

    def analyze_from_paper(
        self,
        paper: BasePaper,
        pdf_path: Optional[str] = None
    ) -> PaperAnalysis:
        """
        分析任意来源的论文（通用接口）

        Args:
            paper: BasePaper对象（ArxivPaper、SciHubPaper等）
            pdf_path: PDF文件路径（如果已下载）

        Returns:
            PaperAnalysis对象
        """
        # 生成paper_id（如果没有）
        paper_id = paper.paper_id or hashlib.md5(paper.title.encode()).hexdigest()[:8]

        # 获取发布日期
        published = ""
        if paper.published_date:
            published = paper.published_date
        elif hasattr(paper, 'year') and paper.year:
            published = str(paper.year)

        # 获取DOI
        doi = ""
        if hasattr(paper, 'doi') and paper.doi:
            doi = paper.doi

        # 获取来源
        from ..apis.paper_base import get_paper_source
        source = get_paper_source(paper)

        analysis = PaperAnalysis(
            title=paper.title,
            authors=paper.authors_str,
            published=published,
            paper_id=paper_id,
            paper_url=paper.paper_url,
            doi=doi,
            source=source
        )

        # 如果是arXiv论文，保留原有字段
        if isinstance(paper, ArxivPaper):
            analysis.arxiv_id = paper.arxiv_id
            analysis.arxiv_url = paper.arxiv_url

        # 如果有PDF文件，分析全文
        if pdf_path and Path(pdf_path).exists():
            return self.analyze_from_pdf(pdf_path, analysis)

        # 否则只分析摘要
        return self.analyze_abstract(paper.summary, analysis)

    def analyze_from_pdf(
        self,
        pdf_path: str,
        base_analysis: Optional[PaperAnalysis] = None
    ) -> PaperAnalysis:
        """
        从PDF文件分析论文

        Args:
            pdf_path: PDF文件路径
            base_analysis: 基础分析对象（如果已存在）

        Returns:
            PaperAnalysis对象
        """
        if base_analysis is None:
            base_analysis = PaperAnalysis()

        with PDFParser(pdf_path) as parser:
            # 更新元数据
            metadata = parser.extract_metadata()
            if not base_analysis.title:
                base_analysis.title = metadata.get("title", "")
            if not base_analysis.authors:
                base_analysis.authors = metadata.get("author", "")

            # 决定分析内容
            if self.analyze_full_text:
                content = self._get_content_for_analysis(parser)
                base_analysis.analysis_type = "full"
            else:
                content = parser.extract_abstract()
                base_analysis.analysis_type = "abstract"

            # 执行分析
            analysis = self._analyze_content(content, base_analysis)

            # 提取参考文献目录（用于后续挖掘，不包含在报告中）
            if self.analyze_full_text:
                analysis.references = self._extract_references_from_pdf(parser)

            return analysis

    def analyze_abstract(
        self,
        abstract: str,
        base_analysis: Optional[PaperAnalysis] = None
    ) -> PaperAnalysis:
        """
        分析论文摘要

        Args:
            abstract: 论文摘要
            base_analysis: 基础分析对象

        Returns:
            PaperAnalysis对象
        """
        if base_analysis is None:
            base_analysis = PaperAnalysis()

        base_analysis.analysis_type = "abstract"

        # 使用摘要分析Prompt
        prompt = self.prompts.get(
            "abstract_analysis_prompt",
            title=base_analysis.title,
            authors=base_analysis.authors,
            abstract=abstract
        )

        try:
            result = self.llm_client.chat([{
                "role": "system",
                "content": "你是一个专业的学术研究助手，擅长分析和总结学术论文。"
            }, {
                "role": "user",
                "content": prompt
            }])

            base_analysis.summary = result
            base_analysis.full_analysis = result
        except Exception as e:
            base_analysis.summary = f"分析失败: {str(e)}"

        return base_analysis

    def _get_content_for_analysis(self, parser: PDFParser) -> str:
        """
        获取用于分析的内容

        Args:
            parser: PDF解析器

        Returns:
            文本内容
        """
        if self.max_pages > 0:
            return parser.extract_pages(0, self.max_pages)
        else:
            return parser.extract_text()

    def _analyze_content(
        self,
        content: str,
        analysis: PaperAnalysis
    ) -> PaperAnalysis:
        """
        分析论文内容

        Args:
            content: 论文内容
            analysis: 基础分析对象

        Returns:
            完整的PaperAnalysis对象
        """
        # 使用完整分析Prompt
        prompt = self.prompts.get(
            "analysis_prompt",
            content=content[:15000],  # 限制长度避免超token
            title=analysis.title,
            authors=analysis.authors,
            published=analysis.published
        )

        try:
            result = self.llm_client.chat([{
                "role": "system",
                "content": "你是一个专业的学术研究助手，擅长分析和总结学术论文。"
            }, {
                "role": "user",
                "content": prompt
            }])

            analysis.full_analysis = result
            analysis.summary = result

            # 尝试解析结构化信息
            self._parse_analysis_result(result, analysis)

        except Exception as e:
            analysis.summary = f"分析失败: {str(e)}"
            analysis.full_analysis = f"分析失败: {str(e)}"

        return analysis

    def _parse_analysis_result(self, result: str, analysis: PaperAnalysis) -> None:
        """
        解析LLM分析结果

        Args:
            result: LLM返回的完整分析结果
            analysis: 要更新的分析对象
        """
        import re

        # 尝试提取各个部分
        sections = {
            "研究领域": "category",
            "核心问题": "core_problem",
            "主要创新点": "contributions",
            "方法概述": "method_overview",
            "实验结果": "experimental_results",
            "局限性": "limitations"
        }

        for section_name, field_name in sections.items():
            pattern = rf"(?:##\s*)?{section_name}[:：]\s*\n(.*?)(?=\n\s*(?:##|###|[0-9]+\.|$))"
            match = re.search(pattern, result, re.DOTALL)

            if match:
                content = match.group(1).strip()

                if field_name == "contributions":
                    # 解析创新点列表
                    contributions = re.findall(r"^\d+[\.\、]\s*(.+)$", content, re.MULTILINE)
                    analysis.contributions = contributions or [content]
                else:
                    setattr(analysis, field_name, content)

    def _extract_references_from_pdf(self, parser: PDFParser) -> List[str]:
        """
        从PDF中提取参考文献目录（用于后续挖掘）

        Args:
            parser: PDF解析器

        Returns:
            参考文献标题列表
        """
        import re

        try:
            # 获取完整文本
            full_text = parser.extract_text()

            # 尝试找到参考文献部分
            # 常见的参考文献标题模式
            ref_patterns = [
                r'\n\s*References\s*\n',
                r'\n\s*参考文献\s*\n',
                r'\n\s*Bibliography\s*\n',
                r'\n\s*参考书目\s*\n'
            ]

            ref_start = -1
            for pattern in ref_patterns:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    ref_start = match.start()
                    break

            if ref_start == -1:
                # 未找到明确的参考文献部分，尝试从最后提取
                ref_section = full_text[-30000:]  # 取最后3万字
            else:
                ref_section = full_text[ref_start:]

            # 使用LLM提取参考文献标题列表
            prompt = f"""请从以下参考文献目录中提取所有论文的标题。

参考文献目录：
{ref_section[:10000]}

请只返回论文标题列表，每行一个标题，不要包含任何其他解释文字。"""

            result = self.llm_client.chat([
                {"role": "system", "content": "你是一个学术文献提取专家。"},
                {"role": "user", "content": prompt}
            ])

            # 解析结果
            references = []
            for line in result.split("\n"):
                line = line.strip()
                # 跳过空行和纯数字编号
                if line and len(line) > 10 and not re.match(r'^\d+[\.\)]?\s*$', line):
                    # 移除开头的数字编号
                    clean_ref = re.sub(r'^[\[\(]?\d+[\]\)]?[\.\)]?\s*', '', line)
                    if len(clean_ref) > 10:
                        references.append(clean_ref[:200])  # 限制长度

            return references[:50]  # 最多返回50篇

        except Exception as e:
            # 提取失败，返回空列表
            return []

    def classify_paper(self, title: str, abstract: str) -> str:
        """
        对论文进行分类

        Args:
            title: 论文标题
            abstract: 论文摘要

        Returns:
            分类结果
        """
        prompt = self.prompts.get(
            "classification_prompt",
            title=title,
            abstract=abstract
        )

        try:
            return self.llm_client.classify_paper(title, abstract, prompt)
        except Exception as e:
            return f"分类失败: {str(e)}"

    def extract_contributions(self, content: str) -> List[str]:
        """
        提取论文创新点

        Args:
            content: 论文内容

        Returns:
            创新点列表
        """
        prompt = self.prompts.get("contributions_prompt", content=content[:10000])

        try:
            result = self.llm_client.extract_contributions(content, prompt)
            # 解析列表
            contributions = []
            for line in result.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    contributions.append(line)
            return contributions or [result]
        except Exception as e:
            return [f"提取失败: {str(e)}"]

    def batch_analyze_papers(
        self,
        papers: List[ArxivPaper],
        show_progress: bool = True
    ) -> List[PaperAnalysis]:
        """
        批量分析论文

        Args:
            papers: ArxivPaper列表
            show_progress: 是否显示进度

        Returns:
            PaperAnalysis列表
        """
        results = []

        for i, paper in enumerate(papers):
            if show_progress:
                print(f"正在分析第 {i+1}/{len(papers)} 篇论文: {paper.title[:50]}...")

            analysis = self.analyze_from_arxiv(paper)
            results.append(analysis)

        return results


def analyze_paper_from_pdf(
    pdf_path: str,
    llm_client: LLMClient,
    analyze_full_text: bool = True
) -> PaperAnalysis:
    """
    便捷函数：分析PDF论文

    Args:
        pdf_path: PDF文件路径
        llm_client: LLM客户端
        analyze_full_text: 是否分析全文

    Returns:
        PaperAnalysis对象
    """
    analyzer = PaperAnalyzer(llm_client, analyze_full_text)
    return analyzer.analyze_from_pdf(pdf_path)


def analyze_paper_from_arxiv(
    paper: ArxivPaper,
    llm_client: LLMClient,
    pdf_path: Optional[str] = None
) -> PaperAnalysis:
    """
    便捷函数：分析arXiv论文

    Args:
        paper: ArxivPaper对象
        llm_client: LLM客户端
        pdf_path: PDF文件路径（可选）

    Returns:
        PaperAnalysis对象
    """
    analyzer = PaperAnalyzer(llm_client)
    return analyzer.analyze_from_arxiv(paper, pdf_path)
