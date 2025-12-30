"""
配置管理模块
加载和管理YAML配置文件
"""

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径，默认为 config/config.yaml
        """
        if config_path is None:
            # 查找项目根目录（包含config文件夹的目录）
            current_path = Path(__file__).resolve()
            for parent in [current_path, current_path.parent, current_path.parent.parent]:
                project_root = parent
                if (project_root / "config" / "config.yaml").exists():
                    break
            else:
                # 如果找不到，使用当前工作目录
                project_root = Path.cwd()

            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path).resolve()
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 解析环境变量
        config = self._parse_env_vars(config)
        return config

    def _parse_env_vars(self, config: Any) -> Any:
        """
        递归解析配置中的环境变量
        支持 ${VAR_NAME} 格式，同时自动覆盖 LLM 相关的环境变量
        """
        if isinstance(config, dict):
            parsed = {}
            for k, v in config.items():
                # LLM配置优先从环境变量读取
                if k == "api_key" and os.getenv("LLM_API_KEY"):
                    parsed[k] = os.getenv("LLM_API_KEY")
                elif k == "base_url" and os.getenv("LLM_BASE_URL"):
                    parsed[k] = os.getenv("LLM_BASE_URL")
                elif k == "model" and os.getenv("LLM_MODEL"):
                    parsed[k] = os.getenv("LLM_MODEL")
                else:
                    parsed[k] = self._parse_env_vars(v)
            return parsed
        elif isinstance(config, list):
            return [self._parse_env_vars(item) for item in config]
        elif isinstance(config, str):
            # 匹配 ${VAR_NAME} 格式
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, config)

            for var_name in matches:
                env_value = os.getenv(var_name)
                if env_value is not None:
                    config = config.replace(f"${{{var_name}}}", env_value)

            return config
        else:
            return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        支持点号分隔的嵌套键，如 "llm.api_key"
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置节"""
        return self._config.get(section, {})

    @property
    def llm(self) -> Dict[str, Any]:
        """LLM配置"""
        return self._config.get("llm", {})

    @property
    def search(self) -> Dict[str, Any]:
        """搜索配置"""
        return self._config.get("search", {})

    @property
    def pdf(self) -> Dict[str, Any]:
        """PDF配置"""
        return self._config.get("pdf", {})

    @property
    def analysis(self) -> Dict[str, Any]:
        """分析配置"""
        return self._config.get("analysis", {})

    @property
    def output(self) -> Dict[str, Any]:
        """输出配置"""
        return self._config.get("output", {})

    def __repr__(self) -> str:
        return f"Config(config_path={self.config_path})"


class PromptConfig:
    """Prompt模板配置"""

    def __init__(self, prompts_path: str = None):
        """
        初始化Prompt配置

        Args:
            prompts_path: Prompt配置文件路径
        """
        if prompts_path is None:
            # 查找项目根目录（包含config文件夹的目录）
            current_path = Path(__file__).resolve()
            for parent in [current_path, current_path.parent, current_path.parent.parent]:
                project_root = parent
                if (project_root / "config" / "prompts.yaml").exists():
                    break
            else:
                # 如果找不到，使用当前工作目录
                project_root = Path.cwd()

            prompts_path = project_root / "config" / "prompts.yaml"

        self.prompts_path = Path(prompts_path).resolve()
        self._prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """加载Prompt配置"""
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompt配置文件不存在: {self.prompts_path}")

        with open(self.prompts_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key: str, **kwargs) -> str:
        """
        获取Prompt模板并格式化

        Args:
            key: Prompt键名
            **kwargs: 格式化参数

        Returns:
            格式化后的Prompt字符串
        """
        template = self._prompts.get(key, "")
        if kwargs:
            return template.format(**kwargs)
        return template

    def __getitem__(self, key: str) -> str:
        return self._prompts.get(key, "")


# 全局配置实例
_global_config: Config = None
_global_prompts: PromptConfig = None


def get_config(config_path: str = None) -> Config:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None or config_path is not None:
        _global_config = Config(config_path)
    return _global_config


def get_prompts(prompts_path: str = None) -> PromptConfig:
    """获取全局Prompt配置实例"""
    global _global_prompts
    if _global_prompts is None or prompts_path is not None:
        _global_prompts = PromptConfig(prompts_path)
    return _global_prompts
