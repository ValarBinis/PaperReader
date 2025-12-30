"""
ReferenceMiner - 参考文献挖掘模块
从论文PDF中提取参考文献，并查找相关arXiv论文
"""

import re
from typing import List, Dict, Optional, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..parsers.pdf_parser import PDFParser
from ..apis.arxiv_api import ArxivAPI


class Reference:
    """参考文献数据类"""

    def __init__(
        self,
        title: str = "",
        authors: List[str] = None,
        year: int = None,
        venue: str = "",
        arxiv_id: Optional[str] = None
    ):
        self.title = title
        self.authors = authors or []
        self.year = year
        self.venue = venue
        self.arxiv_id = arxiv_id

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "arxiv_id": self.arxiv_id
        }


class ReferenceMiner:
    """
    参考文献挖掘器
    从PDF中提取参考文献，查找arXiv论文
    """

    def __init__(self, arxiv_api: ArxivAPI = None):
        """
        初始化ReferenceMiner

        Args:
            arxiv_api: arXiv API客户端
        """
        self.arxiv_api = arxiv_api or ArxivAPI()
        self.visited: Set[str] = set()

    def extract_references(self, pdf_path: str) -> List[Reference]:
        """
        从PDF提取参考文献

        Args:
            pdf_path: PDF文件路径

        Returns:
            参考文献列表
        """
        with PDFParser(pdf_path) as parser:
            text = parser.extract_text()

        references = []

        # 查找参考文献部分
        ref_section = self._extract_reference_section(text)

        if not ref_section:
            return references

        # 解析参考文献条目
        ref_entries = self._split_references(ref_section)

        for entry in ref_entries:
            ref = self._parse_reference(entry)
            if ref and ref.title:
                references.append(ref)

        return references

    def _extract_reference_section(self, text: str) -> Optional[str]:
        """
        提取参考文献部分

        Args:
            text: PDF文本

        Returns:
            参考文献部分文本
        """
        # 常见的参考文献标题模式
        patterns = [
            r'(?:References|参考文献|Bibliography)\s*:?\s*\n(.*?)(?=\n\s*(Appendix|Acknowledgments|附录|致谢|$))',
            r'(?:References|参考文献)\s*\n(.*?)$',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _split_references(self, ref_section: str) -> List[str]:
        """
        分割参考文献条目

        Args:
            ref_section: 参考文献部分文本

        Returns:
            参考文献条目列表
        """
        # 按数字编号分割
        entries = re.split(r'\n\s*\[\d+\]|\n\s*\d+\.|\n\s*\[\d+\]', ref_section)

        # 过滤空条目
        entries = [e.strip() for e in entries if e.strip() and len(e.strip()) > 20]

        return entries

    def _parse_reference(self, entry: str) -> Optional[Reference]:
        """
        解析单条参考文献

        Args:
            entry: 参考文献条目文本

        Returns:
            Reference对象
        """
        # 提取标题（通常在引号中或单独一行）
        title_match = re.search(r'["“]([^"]+)["”]', entry)
        if title_match:
            title = title_match.group(1)
        else:
            # 尝试提取第一句非作者部分
            lines = entry.split('\n')
            title = lines[0] if lines else ""

            # 移除作者名（通常在开头）
            title = re.sub(r'^[A-Z][a-z]+(?:\s+[A-Z]\.\s+)+', '', title)
            title = re.sub(r'^[A-Z][a-z]+\s+et\s+al\.\s*', '', title, flags=re.IGNORECASE)
            title = title.strip('. ')

        # 提取年份
        year_match = re.search(r'\b(19|20)\d{2}\b', entry)
        year = int(year_match.group(0)) if year_match else None

        # 提取作者
        authors = self._extract_authors(entry)

        # 提取发表场所
        venue_match = re.search(r'(?:Proceedings|Journal|Conference|arXiv)[^\n,]+', entry, re.IGNORECASE)
        venue = venue_match.group(0) if venue_match else ""

        # 检查arXiv ID
        arxiv_match = re.search(r'arXiv:\s*(\d+\.\d+)', entry, re.IGNORECASE)
        arxiv_id = arxiv_match.group(1) if arxiv_match else None

        return Reference(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            arxiv_id=arxiv_id
        )

    def _extract_authors(self, entry: str) -> List[str]:
        """
        提取作者名

        Args:
            entry: 参考文献条目

        Returns:
            作者列表
        """
        # 匹配作者名模式
        patterns = [
            r'([A-Z][a-z]+\s+[A-Z]\.)',  # Name I.
            r'([A-Z][a-z]+\s+et\s+al)',  # Name et al
        ]

        authors = []
        for pattern in patterns:
            matches = re.findall(pattern, entry)
            authors.extend(matches[:5])  # 限制最多5个

        return list(set(authors))  # 去重

    def find_arxiv_papers(
        self,
        references: List[Reference],
        max_results: int = 5
    ) -> Dict[str, str]:
        """
        查找参考文献中的arXiv论文

        Args:
            references: 参考文献列表
            max_results: 每个参考文献最多搜索结果数

        Returns:
            {title: arxiv_id} 字典
        """
        arxiv_papers = {}

        for ref in references:
            # 如果已经有arXiv ID，直接添加
            if ref.arxiv_id:
                arxiv_papers[ref.title] = ref.arxiv_id
                continue

            # 按标题搜索arXiv
            try:
                results = self.arxiv_api.search(
                    query=f"all:{ref.title}",
                    max_results=1
                )

                if results and len(results) > 0:
                    # 检查标题相似度
                    if self._title_similarity(ref.title, results[0].title) > 0.7:
                        arxiv_papers[ref.title] = results[0].arxiv_id

            except Exception as e:
                print(f"搜索arXiv失败: {e}")

        return arxiv_papers

    def find_arxiv_papers_from_titles(
        self,
        reference_titles: List[str],
        max_workers: int = 10,
        timeout: float = 3.0
    ) -> Dict[str, str]:
        """
        从参考文献标题列表中查找arXiv论文（并发搜索）

        Args:
            reference_titles: 参考文献标题列表（LLM提取）
            max_workers: 最大并发数
            timeout: 单个搜索超时时间（秒）

        Returns:
            {title: arxiv_id} 字典
        """
        arxiv_papers = {}
        lock = threading.Lock()

        def search_single_title(title: str) -> Optional[tuple]:
            """搜索单个标题"""
            try:
                results = self.arxiv_api.search(
                    query=f"ti:{title}",  # 使用标题搜索更精确
                    max_results=1,
                    timeout=timeout
                )

                if results and len(results) > 0:
                    # 检查标题相似度
                    if self._title_similarity(title, results[0].title) > 0.7:
                        return (title, results[0].arxiv_id)
            except Exception as e:
                pass  # 静默失败，避免干扰其他搜索
            return None

        # 使用线程池并发搜索
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有搜索任务
            future_to_title = {
                executor.submit(search_single_title, title): title
                for title in reference_titles
            }

            # 收集完成的任务
            for future in as_completed(future_to_title):
                try:
                    result = future.result(timeout=timeout)
                    if result:
                        with lock:
                            arxiv_papers[result[0]] = result[1]
                except Exception:
                    pass  # 超时或错误，跳过

        return arxiv_papers

    def _title_similarity(self, title1: str, title2: str) -> float:
        """
        计算标题相似度（简单实现）

        Args:
            title1: 标题1
            title2: 标题2

        Returns:
            相似度分数 0-1
        """
        title1 = title1.lower()
        title2 = title2.lower()

        # 简单的词重叠计算
        words1 = set(title1.split())
        words2 = set(title2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def recursive_expand(
        self,
        paper_id: str,
        max_depth: int = 2,
        current_depth: int = 0,
        visited: Optional[Set[str]] = None
    ) -> Dict[str, List[str]]:
        """
        递归查找相关论文

        Args:
            paper_id: arXiv论文ID
            max_depth: 最大递归深度
            current_depth: 当前深度
            visited: 已访问论文集合

        Returns:
            {paper_id: [related_paper_ids]} 字典
        """
        if visited is None:
            visited = set()

        if paper_id in visited or current_depth >= max_depth:
            return {}

        visited.add(paper_id)
        result = {paper_id: []}

        # 获取论文信息
        paper = self.arxiv_api.get_by_id(paper_id)
        if not paper:
            return result

        # 这里简化处理：搜索相关论文
        # 实际应该从PDF中提取参考文献
        try:
            related = self.arxiv_api.search(
                query=paper.title,
                max_results=5
            )

            for related_paper in related:
                if related_paper.arxiv_id != paper_id:
                    result[paper_id].append(related_paper.arxiv_id)

                    # 递归查找
                    if current_depth < max_depth - 1:
                        child_result = self.recursive_expand(
                            related_paper.arxiv_id,
                            max_depth,
                            current_depth + 1,
                            visited
                        )
                        result.update(child_result)

        except Exception as e:
            print(f"递归搜索失败: {e}")

        return result


def extract_references(pdf_path: str) -> List[Reference]:
    """
    便捷函数：提取PDF参考文献

    Args:
        pdf_path: PDF文件路径

    Returns:
        参考文献列表
    """
    miner = ReferenceMiner()
    return miner.extract_references(pdf_path)
