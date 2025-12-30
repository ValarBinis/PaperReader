"""
ObsidianRenderer - Obsidian知识库渲染器
生成Obsidian兼容的Markdown文件，支持双链接和知识图谱
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..analyzers.paper_analyzer import PaperAnalysis
from ..graph.citation_graph import CitationGraph


class ObsidianRenderer:
    """
    Obsidian知识库渲染器
    生成支持双链接和知识图谱的Markdown文件
    """

    def __init__(self, vault_path: str):
        """
        初始化Obsidian渲染器

        Args:
            vault_path: Obsidian知识库路径
        """
        self.vault_path = Path(vault_path)
        self.papers_dir = self.vault_path / "papers"
        self.topics_dir = self.vault_path / "topics"
        self.authors_dir = self.vault_path / "authors"

        # 创建目录
        for dir_path in [self.papers_dir, self.topics_dir, self.authors_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def sanitize_filename(self, name: str) -> str:
        """
        清理文件名，移除Obsidian不支持的字符

        Args:
            name: 原始名称

        Returns:
            清理后的文件名
        """
        # Obsidian不支持的特殊字符
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']

        for char in invalid_chars:
            name = name.replace(char, "-")

        # 移除前后空格和点
        name = name.strip(". ")

        # 限制长度
        if len(name) > 200:
            name = name[:200]

        return name

    def generate_paper_note(self, analysis: PaperAnalysis) -> str:
        """
        生成单篇论文的Obsidian笔记

        Args:
            analysis: 论文分析结果

        Returns:
            Markdown内容
        """
        lines = []

        # Frontmatter (YAML)
        lines.append("---")
        lines.append("tags:")
        lines.append(f"  - paper")
        if analysis.category:
            category_tag = analysis.category.lower().replace(" ", "-")
            lines.append(f"  - {category_tag}")
        lines.append(f"arxiv_id: {analysis.arxiv_id}")
        lines.append(f"published: {analysis.published}")
        lines.append(f"authors: {analysis.authors}")
        lines.append(f"category: {analysis.category}")
        lines.append("---")
        lines.append("")

        # 标题
        lines.append(f"# {analysis.title}")
        lines.append("")

        # 基本信息
        lines.append("## 基本信息")
        lines.append(f"- **arXiv ID:** [{analysis.arxiv_id}]({analysis.arxiv_url})")
        lines.append(f"- **作者:** {self._format_authors(analysis.authors)}")
        lines.append(f"- **发布时间:** {analysis.published}")
        lines.append(f"- **分类:** {analysis.category}")
        lines.append("")

        # 核心问题
        if analysis.core_problem:
            lines.append("## 核心问题")
            lines.append(analysis.core_problem)
            lines.append("")

        # 创新点
        if analysis.contributions:
            lines.append("## 主要创新点")
            for i, contribution in enumerate(analysis.contributions, 1):
                lines.append(f"{i}. {contribution}")
            lines.append("")

        # 方法概述
        if analysis.method_overview:
            lines.append("## 方法概述")
            lines.append(analysis.method_overview)
            lines.append("")

        # 实验结果
        if analysis.experimental_results:
            lines.append("## 实验结果")
            lines.append(analysis.experimental_results)
            lines.append("")

        # 局限性
        if analysis.limitations:
            lines.append("## 局限性")
            lines.append(analysis.limitations)
            lines.append("")

        # 完整摘要（折叠）
        if analysis.summary and len(analysis.summary) > 100:
            lines.append("<details>")
            lines.append("<summary>详细摘要</summary>")
            lines.append("")
            lines.append(analysis.summary)
            lines.append("")
            lines.append("</details>")

        return "\n".join(lines)

    def _format_authors(self, authors: str) -> str:
        """
        格式化作者为Obsidian链接

        Args:
            authors: 作者字符串

        Returns:
            格式化后的作者字符串
        """
        if isinstance(authors, str):
            author_list = [a.strip() for a in authors.split(",")]
        else:
            author_list = authors

        formatted = []
        for author in author_list[:5]:  # 最多显示5个
            author_name = self.sanitize_filename(author)
            formatted.append(f"[[{author_name}]]")

        if len(author_list) > 5:
            formatted.append("等")

        return ", ".join(formatted)

    def save_paper_note(self, analysis: PaperAnalysis) -> str:
        """
        保存论文笔记到文件

        Args:
            analysis: 论文分析结果

        Returns:
            保存的文件路径
        """
        # 生成文件名
        year = analysis.published[:4] if analysis.published else "0000"
        title_short = analysis.title[:80]
        filename = f"{year}-{self.sanitize_filename(title_short)}.md"
        file_path = self.papers_dir / filename

        # 生成内容
        content = self.generate_paper_note(analysis)

        # 保存
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return str(file_path)

    def generate_topic_index(
        self,
        topic: str,
        papers: List[PaperAnalysis],
        description: str = ""
    ) -> str:
        """
        生成主题索引页

        Args:
            topic: 主题名称
            papers: 相关论文列表
            description: 主题描述

        Returns:
            Markdown内容
        """
        lines = []

        # Frontmatter
        lines.append("---")
        lines.append("tags:")
        lines.append(f"  - topic")
        topic_tag = topic.lower().replace(" ", "-")
        lines.append(f"  - {topic_tag}")
        lines.append("type: topic-index")
        lines.append("---")
        lines.append("")

        # 标题
        lines.append(f"# {topic}")
        lines.append("")

        # 描述
        if description:
            lines.append("## 描述")
            lines.append(description)
            lines.append("")

        # 论文列表
        lines.append("## 相关论文")
        lines.append("")
        lines.append("| 标题 | 作者 | 发布时间 | 分类 |")
        lines.append("|------|------|----------|------|")

        for paper in papers:
            title = paper.title[:50] + "..." if len(paper.title) > 50 else paper.title
            authors = paper.authors[:20] + "..." if len(paper.authors) > 20 else paper.authors

            # 生成论文链接
            year = paper.published[:4] if paper.published else "0000"
            title_short = paper.title[:80]
            note_name = f"{year}-{self.sanitize_filename(title_short)}"
            link = f"[[papers/{note_name}|{paper.title}]]"

            lines.append(f"| {link} | {authors} | {paper.published} | {paper.category} |")

        lines.append("")

        # 统计
        lines.append("## 统计")
        lines.append(f"- **论文数量:** {len(papers)}")
        lines.append(f"- **分类:** {', '.join(set(p.category for p in papers if p.category))}")
        lines.append("")

        return "\n".join(lines)

    def save_topic_index(
        self,
        topic: str,
        papers: List[PaperAnalysis],
        description: str = ""
    ) -> str:
        """
        保存主题索引到文件

        Args:
            topic: 主题名称
            papers: 相关论文列表
            description: 主题描述

        Returns:
            保存的文件路径
        """
        filename = f"{self.sanitize_filename(topic)}.md"
        file_path = self.topics_dir / filename

        content = self.generate_topic_index(topic, papers, description)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return str(file_path)

    def generate_author_note(self, author_name: str, papers: List[PaperAnalysis]) -> str:
        """
        生成作者页面

        Args:
            author_name: 作者名称
            papers: 该作者的论文列表

        Returns:
            Markdown内容
        """
        lines = []

        # Frontmatter
        lines.append("---")
        lines.append("tags:")
        lines.append(f"  - author")
        lines.append(f"name: {author_name}")
        lines.append(f"paper_count: {len(papers)}")
        lines.append("---")
        lines.append("")

        # 标题
        lines.append(f"# {author_name}")
        lines.append("")

        # 论文列表
        lines.append("## 发表论文")
        lines.append("")

        for paper in papers:
            year = paper.published[:4] if paper.published else "0000"
            title_short = paper.title[:80]
            note_name = f"{year}-{self.sanitize_filename(title_short)}"
            link = f"[[papers/{note_name}|{paper.title}]]"

            lines.append(f"- {link} ({paper.published})")

        lines.append("")

        return "\n".join(lines)

    def generate_moc(self, analyses: List[PaperAnalysis], graph: CitationGraph = None) -> str:
        """
        生成MOC (Map of Content) - 知识图谱总览

        Args:
            analyses: 所有论文分析
            graph: 引用关系图

        Returns:
            Markdown内容
        """
        lines = []

        # Frontmatter
        lines.append("---")
        lines.append("tags:")
        lines.append("  - moc")
        lines.append("type: map-of-content")
        lines.append("---")
        lines.append("")

        # 标题
        lines.append(f"# 知识图谱总览")
        lines.append("")
        lines.append(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        # 统计
        lines.append("## 统计信息")
        lines.append(f"- **总论文数:** {len(analyses)}")
        lines.append(f"- **作者数:** {len(set(a.authors for a in analyses))}")
        lines.append("")

        # 按分类统计
        categories = {}
        for analysis in analyses:
            cat = analysis.category or "未分类"
            categories[cat] = categories.get(cat, 0) + 1

        lines.append("## 分类")
        lines.append("")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- [[{cat}]]: {count} 篇")
        lines.append("")

        # 主题索引
        lines.append("## 主题索引")
        lines.append("")

        # 获取所有主题文件
        if self.topics_dir.exists():
            for topic_file in self.topics_dir.glob("*.md"):
                topic_name = topic_file.stem
                lines.append(f"- [[topics/{topic_name}]]")

        lines.append("")

        # 高被引论文
        if graph:
            lines.append("## 高被引论文")
            lines.append("")

            # 计算被引用次数
            cited_counts = {}
            for paper_id in graph.nodes:
                cited_counts[paper_id] = len(graph.get_cited_by(paper_id))

            # 排序
            top_papers = sorted(cited_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            for paper_id, count in top_papers:
                if count > 0:
                    paper = graph.get_paper(paper_id)
                    if paper:
                        lines.append(f"- [[papers/{paper_id}]]: {count} 次被引")

            lines.append("")

        return "\n".join(lines)

    def save_moc(self, analyses: List[PaperAnalysis], graph: CitationGraph = None) -> str:
        """
        保存MOC到文件

        Args:
            analyses: 所有论文分析
            graph: 引用关系图

        Returns:
            保存的文件路径
        """
        file_path = self.vault_path / "_moc.md"

        content = self.generate_moc(analyses, graph)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return str(file_path)

    def export_all_papers(self, analyses: List[PaperAnalysis]) -> List[str]:
        """
        导出所有论文笔记

        Args:
            analyses: 论文分析列表

        Returns:
            保存的文件路径列表
        """
        saved_paths = []

        for analysis in analyses:
            try:
                path = self.save_paper_note(analysis)
                saved_paths.append(path)
            except Exception as e:
                print(f"保存论文笔记失败 ({analysis.title}): {e}")

        return saved_paths


def link_to_obsidian(text: str, link_type: str = "wiki") -> str:
    """
    将普通链接转换为Obsidian格式

    Args:
        text: 包含链接的文本
        link_type: 链接类型 (wiki, markdown)

    Returns:
        转换后的文本
    """
    if link_type == "wiki":
        # 转换为 [[link]] 格式
        text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'[[\2|\1]]', text)

    return text
