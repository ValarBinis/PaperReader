"""
PDF解析器模块
使用PyMuPDF (fitz) 解析PDF文件
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("请安装 PyMuPDF: pip install PyMuPDF")


class PDFParser:
    """
    PDF解析器
    使用PyMuPDF解析PDF文件，提取文本和元数据
    """

    def __init__(self, pdf_path: str):
        """
        初始化PDF解析器

        Args:
            pdf_path: PDF文件路径
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        self.doc = fitz.open(self.pdf_path)
        self._metadata = None
        self._sections = None

    @property
    def page_count(self) -> int:
        """获取PDF页数"""
        return len(self.doc)

    def extract_text(self, pages: Optional[List[int]] = None) -> str:
        """
        提取PDF文本

        Args:
            pages: 指定页码列表（从0开始），None表示全部页面

        Returns:
            提取的文本
        """
        text_parts = []

        if pages is None:
            pages = range(self.page_count)

        for page_num in pages:
            if page_num >= self.page_count:
                break
            page = self.doc[page_num]
            text_parts.append(page.get_text())

        return "\n\n".join(text_parts)

    def extract_metadata(self) -> Dict[str, Any]:
        """
        提取PDF元数据

        Returns:
            元数据字典
        """
        if self._metadata is None:
            metadata = {
                "title": "",
                "author": "",
                "subject": "",
                "keywords": "",
                "creator": "",
                "producer": "",
                "creation_date": "",
                "modification_date": "",
                "page_count": self.page_count
            }

            # 尝试从PDF元数据获取
            pdf_metadata = self.doc.metadata
            metadata["title"] = pdf_metadata.get("title", "")
            metadata["author"] = pdf_metadata.get("author", "")
            metadata["subject"] = pdf_metadata.get("subject", "")
            metadata["keywords"] = pdf_metadata.get("keywords", "")
            metadata["creator"] = pdf_metadata.get("creator", "")
            metadata["producer"] = pdf_metadata.get("producer", "")
            metadata["creation_date"] = pdf_metadata.get("creationDate", "")
            metadata["modification_date"] = pdf_metadata.get("modDate", "")

            # 如果没有标题，尝试从第一页提取
            if not metadata["title"]:
                metadata["title"] = self._extract_title_from_first_page()

            self._metadata = metadata

        return self._metadata

    def _extract_title_from_first_page(self) -> str:
        """
        从第一页提取标题

        Returns:
            标题文本
        """
        if self.page_count == 0:
            return ""

        first_page = self.doc[0]
        text = first_page.get_text()

        # 尝试匹配标题（通常是第一段较大的文字）
        lines = text.split("\n")
        for line in lines[:10]:  # 只看前10行
            line = line.strip()
            if len(line) > 10 and len(line) < 200:  # 标题长度通常在这个范围
                return line

        return ""

    def extract_abstract(self) -> str:
        """
        提取摘要

        Returns:
            摘要文本
        """
        text = self.extract_text()

        # 查找Abstract部分
        abstract_patterns = [
            r"Abstract\s*:?\s*(.*?)(?=\n\s*(Introduction|Keywords|1\.|I\.))",
            r"摘要\s*:?\s*(.*?)(?=\n\s*(关键词|引言|1\.|一、))",
            r"ABSTRACT\s*:?\s*(.*?)(?=\n\s*(INTRODUCTION|KEYWORDS|1\.|I\.))"
        ]

        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                # 清理多余的空白
                abstract = re.sub(r"\s+", " ", abstract)
                return abstract

        # 如果找不到，返回前500字符作为摘要
        return text[:500].strip()

    def extract_sections(self) -> Dict[str, str]:
        """
        提取论文各章节

        Returns:
            章节字典 {章节名: 内容}
        """
        if self._sections is None:
            text = self.extract_text()

            # 常见章节模式
            section_patterns = {
                "introduction": r"(?:Introduction|引言)(.*?)(?=\n\s*(Related Work|Method|Methodology|Methods|Background|2\.))",
                "related_work": r"(?:Related Work|相关工作)(.*?)(?=\n\s*(Method|Methodology|Methods|Background|Preliminaries|2\.|3\.))",
                "method": r"(?:Method|Methodology|Methods|方法)(.*?)(?=\n\s*(Experiments|Experiment|Results|Discussion|Evaluation|3\.|4\.))",
                "experiments": r"(?:Experiments|Experiment|实验)(.*?)(?=\n\s*(Results|Discussion|Conclusion|Conclusions|4\.|5\.))",
                "results": r"(?:Results|结果)(.*?)(?=\n\s*(Discussion|Conclusion|Conclusions|5\.|6\.))",
                "discussion": r"(?:Discussion|讨论)(.*?)(?=\n\s*(Conclusion|Conclusions|Conclusion and Future Work|6\.|7\.))",
                "conclusion": r"(?:Conclusion|Conclusions|结论)(.*?)(?=\n\s*(References|Acknowledgments|参考文献|致谢|$))"
            }

            self._sections = {}
            for section_name, pattern in section_patterns.items():
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    # 清理多余的空白
                    content = re.sub(r"\s+", " ", content)
                    self._sections[section_name] = content

        return self._sections

    def extract_pages(self, start: int, end: int) -> str:
        """
        提取指定范围的页面

        Args:
            start: 起始页（从0开始）
            end: 结束页（不包含）

        Returns:
            提取的文本
        """
        pages = list(range(start, min(end, self.page_count)))
        return self.extract_text(pages)

    def get_page_text(self, page_num: int) -> str:
        """
        获取指定页的文本

        Args:
            page_num: 页码（从0开始）

        Returns:
            页面文本
        """
        if page_num >= self.page_count:
            return ""

        page = self.doc[page_num]
        return page.get_text()

    def extract_by_keywords(
        self,
        keywords: List[str],
        context_lines: int = 5
    ) -> List[Dict[str, Any]]:
        """
        根据关键词提取文本片段

        Args:
            keywords: 关键词列表
            context_lines: 上下文行数

        Returns:
            匹配结果列表
        """
        text = self.extract_text()
        lines = text.split("\n")

        results = []
        for i, line in enumerate(lines):
            for keyword in keywords:
                if keyword.lower() in line.lower():
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = "\n".join(lines[start:end])

                    results.append({
                        "keyword": keyword,
                        "line": i,
                        "context": context
                    })
                    break

        return results

    def extract_references(self) -> List[str]:
        """
        提取参考文献

        Returns:
            参考文献列表
        """
        text = self.extract_text()

        # 查找References部分
        ref_match = re.search(
            r"(?:References|参考文献)\s*:?\s*(.*?)(?=\n\s*$)",
            text,
            re.DOTALL | re.IGNORECASE
        )

        if not ref_match:
            return []

        ref_text = ref_match.group(1)

        # 按行分割，过滤空行
        ref_lines = [
            line.strip()
            for line in ref_text.split("\n")
            if line.strip()
        ]

        return ref_lines

    def get_text_length(self) -> int:
        """
        获取PDF文本总长度

        Returns:
            字符数
        """
        return len(self.extract_text())

    def get_word_count(self) -> int:
        """
        获取PDF字数

        Returns:
            字数
        """
        text = self.extract_text()
        words = text.split()
        return len(words)

    def close(self) -> None:
        """关闭PDF文档"""
        if self.doc:
            self.doc.close()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def __repr__(self) -> str:
        return f"PDFParser(path={self.pdf_path}, pages={self.page_count})"


def parse_pdf(pdf_path: str) -> PDFParser:
    """
    便捷函数：创建PDF解析器

    Args:
        pdf_path: PDF文件路径

    Returns:
        PDFParser实例
    """
    return PDFParser(pdf_path)


def extract_abstract_from_pdf(pdf_path: str) -> str:
    """
    便捷函数：从PDF提取摘要

    Args:
        pdf_path: PDF文件路径

    Returns:
        摘要文本
    """
    with PDFParser(pdf_path) as parser:
        return parser.extract_abstract()


def extract_full_text(pdf_path: str, max_pages: int = 0) -> str:
    """
    便捷函数：从PDF提取全文

    Args:
        pdf_path: PDF文件路径
        max_pages: 最大页数（0表示全部）

    Returns:
        全文文本
    """
    with PDFParser(pdf_path) as parser:
        if max_pages > 0:
            return parser.extract_pages(0, max_pages)
        return parser.extract_text()
