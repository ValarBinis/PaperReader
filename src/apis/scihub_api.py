"""
scihub_api.py - Sci-Hub API模块
使用Sci-Hub搜索和下载论文PDF
"""

import re
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from .paper_base import BasePaper, BasePaperAPI, PaperSource


class SciHubPaper(BasePaper):
    """Sci-Hub论文数据类"""

    def __init__(
        self,
        title: str,
        authors: List[str],
        doi: Optional[str] = None,
        pdf_url: Optional[str] = None,
        summary: str = "",
        year: Optional[int] = None,
        venue: str = "",
        raw_data: Dict[str, Any] = None
    ):
        """
        初始化Sci-Hub论文对象

        Args:
            title: 论文标题
            authors: 作者列表
            doi: DOI标识符
            pdf_url: PDF下载链接
            summary: 论文摘要
            year: 发表年份
            venue: 发表场所
            raw_data: 原始数据
        """
        self._title = title
        self._authors = authors
        self._doi = doi
        self._pdf_url = pdf_url
        self._summary = summary
        self._year = year
        self._venue = venue
        self._raw_data = raw_data or {}

    @property
    def title(self) -> str:
        """论文标题"""
        return self._title

    @property
    def authors(self) -> List[str]:
        """作者列表"""
        return self._authors

    @property
    def summary(self) -> str:
        """论文摘要"""
        return self._summary

    @property
    def pdf_url(self) -> str:
        """PDF下载链接"""
        return self._pdf_url or ""

    @property
    def doi(self) -> Optional[str]:
        """DOI标识符"""
        return self._doi

    @property
    def year(self) -> Optional[int]:
        """发表年份"""
        return self._year

    @property
    def venue(self) -> str:
        """发表场所"""
        return self._venue

    @property
    def published_date(self) -> Optional[str]:
        """发布日期"""
        if self._year:
            return str(self._year)
        return None

    @property
    def paper_id(self) -> str:
        """论文唯一标识符（使用DOI）"""
        return self._doi or ""

    @property
    def paper_url(self) -> str:
        """论文页面URL（通过DOI构建）"""
        if self._doi:
            return f"https://doi.org/{self._doi}"
        return ""

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
            "doi": self.doi,
            "year": self.year,
            "venue": self.venue,
            "pdf_url": self.pdf_url,
            "paper_url": self.paper_url,
            "source": PaperSource.SCIHUB,
            "raw_data": self._raw_data
        }

    def __repr__(self) -> str:
        return f"SciHubPaper(doi='{self.doi}', title='{self.title[:50]}...')"


class SciHubAPI(BasePaperAPI):
    """
    Sci-Hub API客户端
    用于通过DOI搜索和下载论文PDF
    """

    # 可用的Sci-Hub域名列表（按优先级排序）
    AVAILABLE_DOMAINS = [
        "https://sci-hub.se",
        "https://sci-hub.st",
        "https://sci-hub.ru",
        "https://sci-hub.wf",
        "https://sci-hub.mksa.top",
        "https://sci-hub.at",
        "https://sci-hub.do",
    ]

    def __init__(
        self,
        base_url: str = "",
        timeout: int = 60,
        max_retries: int = 3,
        domain_cache_file: Optional[str] = None
    ):
        """
        初始化Sci-Hub API客户端

        Args:
            base_url: Sci-Hub基础URL（留空则自动检测）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            domain_cache_file: 域名缓存文件路径
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self._base_url = base_url
        self._domain_cache_file = domain_cache_file or self._get_default_cache_path()

        # 如果没有指定base_url，尝试从缓存加载或自动检测
        if not base_url:
            self._base_url = self._load_or_detect_domain()

        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _get_default_cache_path(self) -> str:
        """获取默认域名缓存文件路径"""
        cache_dir = Path.home() / ".paperreader"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir / "scihub_domain.txt")

    def _load_or_detect_domain(self) -> str:
        """
        从缓存加载域名或自动检测可用域名

        Returns:
            可用的Sci-Hub域名
        """
        # 尝试从缓存加载
        if Path(self._domain_cache_file).exists():
            try:
                with open(self._domain_cache_file, 'r') as f:
                    cached_domain = f.read().strip()
                    if cached_domain and self._test_domain(cached_domain):
                        return cached_domain
            except Exception:
                pass

        # 自动检测可用域名
        return self._detect_available_domain()

    def _test_domain(self, domain: str) -> bool:
        """
        测试域名是否可用

        Args:
            domain: 域名

        Returns:
            是否可用
        """
        try:
            response = self._session.get(
                domain,
                timeout=10,
                allow_redirects=True
            )
            return response.status_code == 200
        except Exception:
            return False

    def _detect_available_domain(self) -> str:
        """
        自动检测可用的Sci-Hub域名

        Returns:
            第一个可用的域名
        """
        for domain in self.AVAILABLE_DOMAINS:
            if self._test_domain(domain):
                # 缓存可用域名
                try:
                    with open(self._domain_cache_file, 'w') as f:
                        f.write(domain)
                except Exception:
                    pass
                return domain

        # 如果都不可用，返回第一个（可能失效）
        return self.AVAILABLE_DOMAINS[0]

    @property
    def base_url(self) -> str:
        """获取当前使用的Sci-Hub域名"""
        return self._base_url

    def _is_doi(self, identifier: str) -> bool:
        """
        检查是否为DOI格式

        Args:
            identifier: 标识符

        Returns:
            是否为DOI
        """
        doi_pattern = r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$'
        return bool(re.match(doi_pattern, identifier, re.IGNORECASE))

    def _clean_doi(self, doi: str) -> str:
        """
        清理DOI字符串

        Args:
            doi: 原始DOI

        Returns:
            清理后的DOI
        """
        # 移除URL前缀
        doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
        # 移除DOI:前缀
        doi = re.sub(r'^doi:\s*', '', doi, flags=re.IGNORECASE)
        return doi.strip()

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        **kwargs
    ) -> List[SciHubPaper]:
        """
        搜索论文

        Sci-Hub主要用于DOI查询，如果输入标题会先尝试查找DOI

        Args:
            query: 搜索关键词（DOI或标题）
            max_results: 最大结果数（对于DOI查询始终为1）
            **kwargs: 其他参数（如use_crossref）

        Returns:
            SciHubPaper列表
        """
        use_crossref = kwargs.get('use_crossref', True)

        # 清理查询
        cleaned_query = query.strip()

        # 如果是DOI，直接查询
        if self._is_doi(cleaned_query):
            return self._fetch_by_doi(cleaned_query)

        # 如果是标题，尝试通过CrossRef查找DOI
        if use_crossref:
            doi = self._find_doi_by_title(cleaned_query)
            if doi:
                return self._fetch_by_doi(doi)

        # 无法找到DOI，返回空列表
        return []

    def _fetch_by_doi(self, doi: str) -> List[SciHubPaper]:
        """
        通过DOI从Sci-Hub获取论文

        Args:
            doi: DOI标识符

        Returns:
            SciHubPaper列表
        """
        doi = self._clean_doi(doi)

        for attempt in range(self.max_retries):
            try:
                # 构建Sci-Hub URL
                url = f"{self._base_url}/{doi}"

                response = self._session.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True
                )
                response.raise_for_status()

                # 解析页面
                soup = BeautifulSoup(response.content, 'html.parser')

                # 提取PDF链接
                pdf_url = self._extract_pdf_url(soup, response.text)

                # 提取标题
                title = self._extract_title(soup)

                # 提取作者
                authors = self._extract_authors(soup)

                # 提取摘要
                summary = self._extract_summary(soup)

                # 提取年份
                year = self._extract_year(soup)

                paper = SciHubPaper(
                    title=title or f"Paper with DOI: {doi}",
                    authors=authors,
                    doi=doi,
                    pdf_url=pdf_url,
                    summary=summary,
                    year=year
                )

                return [paper]

            except Exception as e:
                if attempt == self.max_retries - 1:
                    # 最后一次尝试失败，尝试切换域名
                    self._base_url = self._detect_available_domain()
                time.sleep(1)

        return []

    def _extract_pdf_url(self, soup: BeautifulSoup, html: str) -> Optional[str]:
        """
        从Sci-Hub页面提取PDF链接

        Args:
            soup: BeautifulSoup对象
            html: 原始HTML

        Returns:
            PDF链接
        """
        # 方法1: 查找embed标签
        embed = soup.find('embed', {'type': 'application/pdf'})
        if embed and embed.get('src'):
            pdf_url = embed['src']
            if pdf_url.startswith('//'):
                pdf_url = 'https:' + pdf_url
            elif pdf_url.startswith('/'):
                pdf_url = self._base_url + pdf_url
            return pdf_url

        # 方法2: 查找iframe标签
        iframe = soup.find('iframe')
        if iframe and iframe.get('src'):
            pdf_url = iframe['src']
            if pdf_url.startswith('//'):
                pdf_url = 'https:' + pdf_url
            return pdf_url

        # 方法3: 正则匹配PDF链接
        pdf_patterns = [
            r'location\.href\s*=\s*["\']([^"\']+\.pdf)["\']',
            r'["\'](https?://[^\s"\']+\.pdf)["\']',
            r'src\s*=\s*["\']([^"\']+\.pdf)["\']',
        ]

        for pattern in pdf_patterns:
            match = re.search(pattern, html)
            if match:
                pdf_url = match.group(1)
                if pdf_url.startswith('//'):
                    pdf_url = 'https:' + pdf_url
                return pdf_url

        return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """提取标题"""
        # 尝试多种方式提取标题
        selectors = [
            'h1',
            '.title',
            '#title',
            'meta[property="og:title"]'
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    title = element.get('content', '')
                else:
                    title = element.get_text(strip=True)
                if title:
                    return title

        return ""

    def _extract_authors(self, soup: BeautifulSoup) -> List[str]:
        """提取作者"""
        authors = []

        # 尝试从meta标签提取
        meta_authors = soup.find_all('meta', {'name': 'citation_author'})
        if meta_authors:
            authors = [author.get('content', '') for author in meta_authors if author.get('content')]
        else:
            # 尝试从页面内容提取
            authors_section = soup.find('div', {'class': 'authors'})
            if authors_section:
                author_elements = authors_section.find_all('a')
                authors = [a.get_text(strip=True) for a in author_elements]

        return authors

    def _extract_summary(self, soup: BeautifulSoup) -> str:
        """提取摘要"""
        # 尝试从meta标签提取
        meta_abstract = soup.find('meta', {'name': 'citation_abstract'})
        if meta_abstract:
            return meta_abstract.get('content', '')

        # 尝试从页面内容提取
        abstract_section = soup.find('div', {'class': 'abstract'})
        if abstract_section:
            return abstract_section.get_text(strip=True)

        return ""

    def _extract_year(self, soup: BeautifulSoup) -> Optional[int]:
        """提取年份"""
        # 尝试从meta标签提取
        meta_date = soup.find('meta', {'name': 'citation_publication_date'})
        if meta_date:
            date_str = meta_date.get('content', '')
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                return int(year_match.group(0))

        return None

    def _find_doi_by_title(self, title: str) -> Optional[str]:
        """
        通过CrossRef API根据标题查找DOI

        Args:
            title: 论文标题

        Returns:
            DOI或None
        """
        try:
            # 使用CrossRef API
            url = "https://api.crossref.org/works"
            params = {
                'query.title': title,
                'rows': 1,
                'select': 'DOI,title,author,published-print'
            }

            response = requests.get(
                url,
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            if data.get('status') == 'ok' and data.get('message', {}).get('items'):
                item = data['message']['items'][0]
                doi = item.get('DOI')
                if doi:
                    return doi

        except Exception as e:
            pass

        return None

    def get_by_id(self, paper_id: str) -> Optional[SciHubPaper]:
        """
        通过DOI获取论文

        Args:
            paper_id: DOI

        Returns:
            SciHubPaper对象或None
        """
        results = self.search(paper_id, use_crossref=False)
        return results[0] if results else None

    def download_pdf(
        self,
        paper: SciHubPaper,
        save_path: str,
        timeout: int = 120
    ) -> bool:
        """
        下载论文PDF

        Args:
            paper: SciHubPaper对象
            save_path: 保存路径
            timeout: 下载超时时间（秒）

        Returns:
            是否下载成功
        """
        if not paper.pdf_url:
            return False

        try:
            # 确保目录存在
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)

            # 下载PDF
            response = self._session.get(
                paper.pdf_url,
                timeout=timeout,
                stream=True
            )
            response.raise_for_status()

            # 保存文件
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # 验证PDF文件
            if self._is_valid_pdf(save_path):
                return True
            else:
                # 下载的文件不是有效PDF，删除
                Path(save_path).unlink(missing_ok=True)
                return False

        except Exception as e:
            # 下载失败，清理文件
            Path(save_path).unlink(missing_ok=True)
            return False

    def _is_valid_pdf(self, file_path: str) -> bool:
        """
        验证文件是否为有效PDF

        Args:
            file_path: 文件路径

        Returns:
            是否为有效PDF
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                return header == b'%PDF'
        except Exception:
            return False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SciHubAPI":
        """
        从配置创建实例

        Args:
            config: 配置字典

        Returns:
            SciHubAPI实例
        """
        return cls(
            base_url=config.get("base_url", ""),
            timeout=config.get("timeout", 60),
            max_retries=config.get("max_retries", 3)
        )


def search_papers_by_doi(doi: str, timeout: int = 60) -> Optional[SciHubPaper]:
    """
    便捷函数：通过DOI搜索论文

    Args:
        doi: DOI标识符
        timeout: 超时时间

    Returns:
        SciHubPaper对象
    """
    api = SciHubAPI(timeout=timeout)
    results = api.search(doi)
    return results[0] if results else None


def download_paper_by_doi(doi: str, save_path: str, timeout: int = 120) -> bool:
    """
    便捷函数：通过DOI下载论文

    Args:
        doi: DOI标识符
        save_path: 保存路径
        timeout: 超时时间

    Returns:
        是否下载成功
    """
    api = SciHubAPI(timeout=timeout)
    paper = api.get_by_id(doi)

    if paper:
        return api.download_pdf(paper, save_path, timeout)

    return False
