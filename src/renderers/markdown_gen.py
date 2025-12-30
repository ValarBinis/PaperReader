"""
Markdown生成器模块
生成论文索引和报告
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from ..analyzers.paper_analyzer import PaperAnalysis
from ..apis.arxiv_api import ArxivPaper


class MarkdownGenerator:
    """
    Markdown生成器
    生成论文索引和分析报告
    """

    def __init__(self, include_full_summary: bool = True):
        """
        初始化Markdown生成器

        Args:
            include_full_summary: 是否包含完整摘要
        """
        self.include_full_summary = include_full_summary

    def generate_index(
        self,
        analyses: List[PaperAnalysis],
        title: str = "论文索引",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        生成论文索引Markdown

        Args:
            analyses: 论文分析列表
            title: 索引标题
            metadata: 元数据信息

        Returns:
            Markdown字符串
        """
        lines = []

        # 标题
        lines.append(f"# {title}\n")
        lines.append(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**论文数量:** {len(analyses)}\n")

        # 元数据
        if metadata:
            lines.append("## 搜索配置\n")
            for key, value in metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        # 目录
        lines.append("## 目录\n")
        categories = self._categorize_analyses(analyses)
        for category, cat_analyses in categories.items():
            lines.append(f"- [{category}](#{category.lower().replace(' ', '-')}) ({len(cat_analyses)})")
        lines.append("")

        # 按分类生成内容
        for category, cat_analyses in categories.items():
            lines.append(f"## {category}\n")
            lines.append(f"**数量:** {len(cat_analyses)} 篇\n")

            for analysis in cat_analyses:
                lines.extend(self._format_paper_entry(analysis))
                lines.append("")

        return "\n".join(lines)

    def _format_paper_entry(self, analysis: PaperAnalysis) -> List[str]:
        """
        格式化单条论文记录

        Args:
            analysis: 论文分析结果

        Returns:
            Markdown行列表
        """
        lines = []

        # 标题
        lines.append(f"### {analysis.title}")

        # 基本信息
        lines.append("")
        lines.append("**基本信息**")
        lines.append(f"- **作者:** {analysis.authors}")
        lines.append(f"- **发布时间:** {analysis.published}")
        if analysis.arxiv_id:
            lines.append(f"- **arXiv ID:** [{analysis.arxiv_id}]({analysis.arxiv_url})")
        lines.append(f"- **分类:** {analysis.category}")

        # 核心问题
        if analysis.core_problem:
            lines.append("")
            lines.append("**核心问题**")
            lines.append(analysis.core_problem)

        # 创新点
        if analysis.contributions:
            lines.append("")
            lines.append("**主要创新点**")
            for i, contribution in enumerate(analysis.contributions, 1):
                lines.append(f"{i}. {contribution}")

        # 方法概述
        if analysis.method_overview:
            lines.append("")
            lines.append("**方法概述**")
            lines.append(analysis.method_overview)

        # 实验结果
        if analysis.experimental_results:
            lines.append("")
            lines.append("**实验结果**")
            lines.append(analysis.experimental_results)

        # 局限性
        if analysis.limitations:
            lines.append("")
            lines.append("**局限性**")
            lines.append(analysis.limitations)

        # 完整摘要
        if self.include_full_summary and analysis.summary:
            lines.append("")
            lines.append("<details>")
            lines.append("")
            lines.append("**详细摘要**")
            lines.append("")
            lines.append(analysis.summary)
            lines.append("")
            lines.append("</details>")

        return lines

    def _categorize_analyses(
        self,
        analyses: List[PaperAnalysis]
    ) -> Dict[str, List[PaperAnalysis]]:
        """
        按分类组织论文

        Args:
            analyses: 论文分析列表

        Returns:
            分类字典
        """
        categories = defaultdict(list)

        for analysis in analyses:
            category = analysis.category or "未分类"
            categories[category].append(analysis)

        # 按论文数量排序
        sorted_categories = dict(
            sorted(
                categories.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
        )

        return sorted_categories

    def generate_summary_table(
        self,
        analyses: List[PaperAnalysis]
    ) -> str:
        """
        生成论文摘要表格

        Args:
            analyses: 论文分析列表

        Returns:
            Markdown表格字符串
        """
        lines = []

        # 表头
        lines.append("| 标题 | 作者 | 分类 | 发布时间 |")
        lines.append("|------|------|------|----------|")

        # 表格内容
        for analysis in analyses:
            title = analysis.title[:50] + "..." if len(analysis.title) > 50 else analysis.title
            authors = analysis.authors[:20] + "..." if len(analysis.authors) > 20 else analysis.authors

            lines.append(f"| {title} | {authors} | {analysis.category} | {analysis.published} |")

        return "\n".join(lines)

    def generate_contribution_summary(
        self,
        analyses: List[PaperAnalysis]
    ) -> str:
        """
        生成创新点汇总

        Args:
            analyses: 论文分析列表

        Returns:
            Markdown字符串
        """
        lines = []
        lines.append("# 创新点汇总\n")

        for i, analysis in enumerate(analyses, 1):
            if not analysis.contributions:
                continue

            lines.append(f"## {i}. {analysis.title}")
            lines.append("")

            for j, contribution in enumerate(analysis.contributions, 1):
                lines.append(f"{j}. {contribution}")

            lines.append("")

        return "\n".join(lines)

    def generate_daily_report(
        self,
        analyses: List[PaperAnalysis],
        query: str = "",
        date: Optional[datetime] = None
    ) -> str:
        """
        生成日报

        Args:
            analyses: 论文分析列表
            query: 搜索关键词
            date: 日期

        Returns:
            Markdown字符串
        """
        if date is None:
            date = datetime.now()

        lines = []

        # 标题
        lines.append(f"# 论文日报 - {date.strftime('%Y年%m月%d日')}")
        lines.append("")
        lines.append(f"**搜索关键词:** {query}")
        lines.append(f"**论文数量:** {len(analyses)}")
        lines.append("")

        # 统计信息
        categories = self._categorize_analyses(analyses)
        lines.append("## 分类统计\n")
        lines.append("| 分类 | 数量 |")
        lines.append("|------|------|")
        for category, cat_analyses in categories.items():
            lines.append(f"| {category} | {len(cat_analyses)} |")
        lines.append("")

        # 摘要表格
        lines.append("## 论文列表\n")
        lines.append(self.generate_summary_table(analyses))
        lines.append("")

        # 详细内容
        lines.extend(self.generate_index(analyses, "详细内容").split("\n")[4:])  # 跳过标题

        return "\n".join(lines)

    def save_to_file(
        self,
        content: str,
        file_path: str
    ) -> None:
        """
        保存Markdown到文件

        Args:
            content: Markdown内容
            file_path: 文件路径
        """
        from pathlib import Path

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)


def format_arxiv_paper(paper: ArxivPaper) -> str:
    """
    格式化arXiv论文为Markdown

    Args:
        paper: ArxivPaper对象

    Returns:
        Markdown字符串
    """
    lines = []

    lines.append(f"### {paper.title}")
    lines.append("")
    lines.append(f"- **作者:** {paper.authors_str}")
    lines.append(f"- **发布时间:** {paper.published.strftime('%Y-%m-%d')}")
    lines.append(f"- **arXiv ID:** [{paper.arxiv_id}]({paper.arxiv_url})")
    lines.append(f"- **分类:** {', '.join(paper.categories)}")
    lines.append("")
    lines.append("**摘要**")
    lines.append(paper.summary)
    lines.append("")

    return "\n".join(lines)


def generate_papers_list(
    papers: List[ArxivPaper],
    title: str = "论文列表"
) -> str:
    """
    生成论文列表Markdown

    Args:
        papers: ArxivPaper列表
        title: 标题

    Returns:
        Markdown字符串
    """
    lines = []

    lines.append(f"# {title}\n")
    lines.append(f"**数量:** {len(papers)}\n")

    for paper in papers:
        lines.append(format_arxiv_paper(paper))

    return "\n".join(lines)
