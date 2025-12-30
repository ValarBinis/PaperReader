"""
paper_base.py - 论文数据源统一接口
定义论文对象和API的抽象基类，支持多种数据源（arXiv、Sci-Hub等）
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BasePaper(ABC):
    """
    论文数据抽象基类
    所有论文数据类（ArxivPaper、SciHubPaper等）必须实现此接口
    """

    @property
    @abstractmethod
    def title(self) -> str:
        """论文标题"""
        pass

    @property
    @abstractmethod
    def authors(self) -> List[str]:
        """作者列表"""
        pass

    @property
    def authors_str(self) -> str:
        """作者字符串（逗号分隔）"""
        return ", ".join(self.authors)

    @property
    @abstractmethod
    def summary(self) -> str:
        """论文摘要"""
        pass

    @property
    @abstractmethod
    def pdf_url(self) -> str:
        """PDF下载链接"""
        pass

    @property
    def published_date(self) -> Optional[str]:
        """发布日期（可选，某些数据源可能没有）"""
        return None

    @property
    def paper_id(self) -> str:
        """论文唯一标识符（默认为空，子类可重写）"""
        return ""

    @property
    def paper_url(self) -> str:
        """论文页面URL（可选）"""
        return ""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            论文信息字典
        """
        pass

    def __repr__(self) -> str:
        return f"BasePaper(title='{self.title[:50]}...')"


class BasePaperAPI(ABC):
    """
    论文API抽象基类
    所有论文API类（ArxivAPI、SciHubAPI等）必须实现此接口
    """

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        **kwargs
    ) -> List[BasePaper]:
        """
        搜索论文

        Args:
            query: 搜索关键词
            max_results: 最大结果数
            **kwargs: 其他搜索参数

        Returns:
            论文列表
        """
        pass

    @abstractmethod
    def download_pdf(
        self,
        paper: BasePaper,
        save_path: str,
        timeout: int = 120
    ) -> bool:
        """
        下载论文PDF

        Args:
            paper: 论文对象
            save_path: 保存路径
            timeout: 下载超时时间（秒）

        Returns:
            是否下载成功
        """
        pass

    @abstractmethod
    def get_by_id(self, paper_id: str) -> Optional[BasePaper]:
        """
        通过论文ID获取论文

        Args:
            paper_id: 论文ID

        Returns:
            论文对象，如果不存在返回None
        """
        pass

    def search_by_field(
        self,
        query: str,
        field: str = "all",
        max_results: int = 10,
        **kwargs
    ) -> List[BasePaper]:
        """
        按特定字段搜索（默认实现，子类可重写）

        Args:
            query: 搜索关键词
            field: 搜索字段（all/ti/abs/au等）
            max_results: 最大结果数
            **kwargs: 其他参数

        Returns:
            论文列表
        """
        return self.search(query, max_results, **kwargs)


class PaperSource:
    """
    论文数据源枚举
    用于标识论文来源
    """
    ARXIV = "arxiv"
    SCIHUB = "scihub"
    SSRN = "ssrn"
    UNKNOWN = "unknown"


def get_paper_source(paper: BasePaper) -> str:
    """
    获取论文来源

    Args:
        paper: 论文对象

    Returns:
        数据源标识
    """
    class_name = paper.__class__.__name__
    if "Arxiv" in class_name:
        return PaperSource.ARXIV
    elif "SciHub" in class_name:
        return PaperSource.SCIHUB
    elif "SSRN" in class_name:
        return PaperSource.SSRN
    else:
        return PaperSource.UNKNOWN
