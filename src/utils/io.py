"""
文件IO工具模块
提供文件读写、目录操作等工具函数
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


def ensure_dir(path: str | Path) -> Path:
    """
    确保目录存在，不存在则创建

    Args:
        path: 目录路径

    Returns:
        Path对象
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text(file_path: str | Path, encoding: str = "utf-8") -> str:
    """
    读取文本文件

    Args:
        file_path: 文件路径
        encoding: 文件编码

    Returns:
        文件内容字符串
    """
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def write_text(file_path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """
    写入文本文件

    Args:
        file_path: 文件路径
        content: 文件内容
        encoding: 文件编码
    """
    ensure_dir(Path(file_path).parent)
    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)


def read_json(file_path: str | Path) -> Any:
    """
    读取JSON文件

    Args:
        file_path: 文件路径

    Returns:
        JSON数据
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(file_path: str | Path, data: Any, indent: int = 2) -> None:
    """
    写入JSON文件

    Args:
        file_path: 文件路径
        data: 要写入的数据
        indent: 缩进空格数
    """
    ensure_dir(Path(file_path).parent)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def get_file_hash(file_path: str | Path) -> str:
    """
    计算文件的MD5哈希值

    Args:
        file_path: 文件路径

    Returns:
        MD5哈希字符串
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def get_file_size(file_path: str | Path) -> int:
    """
    获取文件大小（字节）

    Args:
        file_path: 文件路径

    Returns:
        文件大小
    """
    return Path(file_path).stat().st_size


def format_size(size_bytes: int) -> str:
    """
    格式化文件大小

    Args:
        size_bytes: 字节数

    Returns:
        格式化后的大小字符串，如 "1.5 MB"
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def list_files(
    directory: str | Path,
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """
    列出目录中的文件

    Args:
        directory: 目录路径
        pattern: 文件匹配模式
        recursive: 是否递归

    Returns:
        文件路径列表
    """
    dir_path = Path(directory)
    if recursive:
        return list(dir_path.rglob(pattern))
    else:
        return list(dir_path.glob(pattern))


def clean_directory(directory: str | Path, pattern: str = "*") -> int:
    """
    清空目录中匹配模式的文件

    Args:
        directory: 目录路径
        pattern: 文件匹配模式

    Returns:
        删除的文件数量
    """
    count = 0
    for file_path in list_files(directory, pattern):
        if file_path.is_file():
            file_path.unlink()
            count += 1
    return count


def generate_filename(
    base_name: str,
    extension: str = "",
    timestamp: bool = True
) -> str:
    """
    生成文件名

    Args:
        base_name: 基础名称
        extension: 扩展名（不含点）
        timestamp: 是否添加时间戳

    Returns:
        生成的文件名
    """
    # 清理文件名中的非法字符
    base_name = "".join(c for c in base_name if c.isalnum() or c in (" ", "-", "_"))
    base_name = base_name.strip()

    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{base_name}_{timestamp_str}"

    if extension:
        if not extension.startswith("."):
            extension = "." + extension
        return f"{base_name}{extension}"
    return base_name


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除非法字符

    Args:
        filename: 原始文件名

    Returns:
        清理后的文件名
    """
    # Windows非法字符
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # 移除前后空格和点
    filename = filename.strip(". ")

    # 限制长度
    if len(filename) > 255:
        name, ext = Path(filename).stem, Path(filename).suffix
        filename = name[:255 - len(ext)] + ext

    return filename


class CacheManager:
    """缓存管理器"""

    def __init__(self, cache_dir: str | Path):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = ensure_dir(cache_dir)

    def get_cache_path(self, key: str, extension: str = ".json") -> Path:
        """
        获取缓存文件路径

        Args:
            key: 缓存键
            extension: 文件扩展名

        Returns:
            缓存文件路径
        """
        # 使用哈希作为文件名避免路径问题
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}{extension}"

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存

        Args:
            key: 缓存键
            default: 默认值

        Returns:
            缓存数据或默认值
        """
        cache_path = self.get_cache_path(key)
        if cache_path.exists():
            try:
                return read_json(cache_path)
            except Exception:
                return default
        return default

    def set(self, key: str, value: Any) -> None:
        """
        设置缓存

        Args:
            key: 缓存键
            value: 缓存数据
        """
        cache_path = self.get_cache_path(key)
        write_json(cache_path, value)

    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            是否存在
        """
        return self.get_cache_path(key).exists()

    def clear(self) -> int:
        """
        清空所有缓存

        Returns:
            删除的文件数量
        """
        return clean_directory(self.cache_dir)


def safe_filename_from_title(title: str, max_length: int = 200) -> str:
    """
    从论文标题生成安全的文件名

    Args:
        title: 论文标题
        max_length: 最大长度

    Returns:
        安全的文件名
    """
    # 移除或替换非法字符
    filename = sanitize_filename(title)

    # 限制长度
    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename
