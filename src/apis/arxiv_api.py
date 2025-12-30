"""
arXiv API模块
使用arxiv库搜索和获取论文
"""

import time
import urllib.parse
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import arxiv
from dateutil import parser as date_parser

from .paper_base import BasePaper, BasePaperAPI, PaperSource


class ArxivPaper(BasePaper):
    """arXiv论文数据类"""

    def __init__(self, result: arxiv.Result):
        """
        初始化论文对象

        Args:
            result: arxiv库的Result对象
        """
        self._result = result

    @property
    def title(self) -> str:
        """论文标题"""
        return self._result.title

    @property
    def authors(self) -> List[str]:
        """作者列表"""
        return [str(author) for author in self._result.authors]

    @property
    def authors_str(self) -> str:
        """作者字符串（逗号分隔）"""
        return ", ".join(self.authors)

    @property
    def summary(self) -> str:
        """论文摘要"""
        return self._result.summary

    @property
    def published(self) -> datetime:
        """发布日期"""
        return self._result.published

    @property
    def updated(self) -> datetime:
        """更新日期"""
        return self._result.updated

    @property
    def arxiv_id(self) -> str:
        """arXiv ID"""
        return self._result.entry_id.split("/")[-1]

    @property
    def arxiv_url(self) -> str:
        """arXiv URL"""
        return self._result.entry_id

    @property
    def pdf_url(self) -> str:
        """PDF下载链接"""
        return self._result.pdf_url

    @property
    def primary_category(self) -> str:
        """主分类"""
        return self._result.primary_category

    @property
    def categories(self) -> List[str]:
        """所有分类"""
        return self._result.categories

    @property
    def comment(self) -> Optional[str]:
        """评论信息"""
        return self._result.comment

    @property
    def journal_ref(self) -> Optional[str]:
        """期刊引用"""
        return self._result.journal_ref

    @property
    def doi(self) -> Optional[str]:
        """DOI"""
        return self._result.doi

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            论文信息字典
        """
        return {
            "title": self.title,
            "authors": self.authors,
            "authors_str": self.authors_str,
            "summary": self.summary,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "arxiv_id": self.arxiv_id,
            "arxiv_url": self.arxiv_url,
            "pdf_url": self.pdf_url,
            "primary_category": self.primary_category,
            "categories": self.categories,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "doi": self.doi,
            "source": PaperSource.ARXIV
        }

    @property
    def paper_id(self) -> str:
        """论文唯一标识符（arXiv ID）"""
        return self.arxiv_id

    @property
    def paper_url(self) -> str:
        """论文页面URL"""
        return self.arxiv_url

    @property
    def published_date(self) -> Optional[str]:
        """发布日期"""
        return self.published.strftime("%Y-%m-%d") if self.published else None


class ArxivAPI(BasePaperAPI):
    """
    arXiv API客户端
    用于搜索和获取arXiv论文
    """

    # 排序方式枚举
    SORT_RELEVANCE = arxiv.SortCriterion.Relevance
    SORT_LAST_UPDATED_DATE = arxiv.SortCriterion.LastUpdatedDate
    SORT_SUBMITTED_DATE = arxiv.SortCriterion.SubmittedDate

    def __init__(
        self,
        max_results: int = 10,
        sort_by: str = "relevance",
        categories: Optional[List[str]] = None,
        search_field: str = "all"
    ):
        """
        初始化arXiv API客户端

        Args:
            max_results: 默认最大结果数
            sort_by: 排序方式 (relevance, date, updated)
            categories: 默认分类过滤
            search_field: 搜索字段 (all=所有字段, ti=标题, abs=摘要)
        """
        self.max_results = max_results
        self.sort_by = self._parse_sort_by(sort_by)
        self.categories = categories or []
        self.search_field = search_field

    def _parse_sort_by(self, sort_by: str) -> arxiv.SortCriterion:
        """解析排序方式"""
        sort_map = {
            "relevance": self.SORT_RELEVANCE,
            "date": self.SORT_SUBMITTED_DATE,
            "updated": self.SORT_LAST_UPDATED_DATE
        }
        return sort_map.get(sort_by, self.SORT_RELEVANCE)

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        sort_by: Optional[str] = None,
        categories: Optional[List[str]] = None,
        days_ago: int = 0,
        timeout: Optional[float] = None
    ) -> List[ArxivPaper]:
        """
        搜索arXiv论文

        Args:
            query: 搜索关键词
            max_results: 最大结果数
            sort_by: 排序方式
            categories: 分类过滤
            days_ago: 时间过滤（最近N天，0表示不限）
            timeout: 超时时间（秒），None表示使用默认值

        Returns:
            ArxivPaper列表
        """
        # 构建搜索查询
        search_query = self._build_query(query, categories, days_ago)

        # 设置排序
        sort_criterion = self._parse_sort_by(sort_by) if sort_by else self.sort_by

        # 创建搜索
        max_results = max_results or self.max_results
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending
        )

        # 执行搜索
        papers = []
        try:
            # 如果指定了超时，使用并发器的超时机制
            if timeout:
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Search timeout after {timeout} seconds")

                # 设置超时信号（仅Unix系统）
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout))
                    for result in search.results():
                        papers.append(ArxivPaper(result))
                    signal.alarm(0)  # 取消超时
                except (AttributeError, ValueError):
                    # Windows不支持SIGALRM，使用基本超时
                    for result in search.results():
                        papers.append(ArxivPaper(result))
            else:
                for result in search.results():
                    papers.append(ArxivPaper(result))
        except TimeoutError:
            pass  # 超时直接返回已获取的结果
        except Exception as e:
            print(f"搜索时出错: {e}")

        return papers

    def _build_query(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        days_ago: int = 0
    ) -> str:
        """
        构建arXiv搜索查询

        Args:
            query: 搜索关键词
            categories: 分类过滤
            days_ago: 时间过滤

        Returns:
            搜索查询字符串
        """
        # URL编码查询词
        encoded_query = urllib.parse.quote(query)

        # 添加分类过滤
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query = f"({cat_query}) AND {self.search_field}:{encoded_query}"
        else:
            query = f"{self.search_field}:{encoded_query}"

        # 添加时间过滤（arXiv API本身不支持，需要后处理）
        # 这里在查询中不做处理，在结果中过滤

        return query

    def get_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        通过arXiv ID获取论文

        Args:
            arxiv_id: arXiv ID (如 "2301.00001")

        Returns:
            ArxivPaper对象，如果不存在返回None
        """
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(search.results())
            return ArxivPaper(result)
        except Exception as e:
            print(f"获取论文 {arxiv_id} 时出错: {e}")
            return None

    def get_recent(
        self,
        categories: List[str],
        max_results: int = 10,
        days_ago: int = 7
    ) -> List[ArxivPaper]:
        """
        获取最近提交的论文

        Args:
            categories: arXiv分类列表
            max_results: 最大结果数
            days_ago: 最近N天

        Returns:
            ArxivPaper列表
        """
        # 构建查询
        cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
        query = f"({cat_query})"

        # 搜索
        papers = self.search(
            query="",
            max_results=max_results * 2,  # 获取更多结果用于时间过滤
            sort_by="date",
            categories=categories
        )

        # 按时间过滤
        cutoff_date = datetime.now() - timedelta(days=days_ago)
        filtered_papers = [
            p for p in papers
            if p.published.replace(tzinfo=None) >= cutoff_date
        ]

        return filtered_papers[:max_results]

    def download_pdf(
        self,
        paper: ArxivPaper,
        save_path: str,
        timeout: int = 120
    ) -> bool:
        """
        下载论文PDF

        Args:
            paper: ArxivPaper对象
            save_path: 保存路径
            timeout: 下载超时时间(秒)

        Returns:
            是否下载成功
        """
        try:
            # 新版arxiv库不再支持slugify参数，先尝试使用官方方法
            # 如果失败，使用requests直接下载
            try:
                import os
                dirpath = os.path.dirname(save_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                paper._result.download_pdf(filename=save_path)
            except TypeError:
                # 旧版本兼容或不支持某些参数，使用直接下载
                import requests
                response = requests.get(paper.pdf_url, timeout=timeout)
                response.raise_for_status()
                import os
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(response.content)
            return True
        except Exception as e:
            print(f"下载PDF失败: {e}")
            return False

    def get_papers_by_keyword(
        self,
        keyword: str,
        max_results: int = 10,
        categories: Optional[List[str]] = None
    ) -> List[ArxivPaper]:
        """
        按关键词搜索论文

        Args:
            keyword: 关键词
            max_results: 最大结果数
            categories: 分类过滤

        Returns:
            ArxivPaper列表
        """
        return self.search(
            query=keyword,
            max_results=max_results,
            categories=categories
        )

    def get_papers_by_author(
        self,
        author_name: str,
        max_results: int = 10
    ) -> List[ArxivPaper]:
        """
        按作者搜索论文

        Args:
            author_name: 作者姓名
            max_results: 最大结果数

        Returns:
            ArxivPaper列表
        """
        query = f"au:{urllib.parse.quote(author_name)}"
        return self.search(query, max_results=max_results)

    @staticmethod
    def get_category_name(category: str) -> str:
        """
        获取分类名称

        Args:
            category: 分类代码 (如 cs.AI)

        Returns:
            分类名称
        """
        category_map = {
            "cs.AI": "Artificial Intelligence",
            "cs.CL": "Computation and Language",
            "cs.CV": "Computer Vision",
            "cs.LG": "Machine Learning",
            "cs.NE": "Neural and Evolutionary Computing",
            "stat.ML": "Machine Learning (Statistics)"
        }
        return category_map.get(category, category)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ArxivAPI":
        """
        从配置创建实例

        Args:
            config: 配置字典

        Returns:
            ArxivAPI实例
        """
        return cls(
            max_results=config.get("max_results", 10),
            sort_by=config.get("sort_by", "relevance"),
            categories=config.get("categories", [])
        )
