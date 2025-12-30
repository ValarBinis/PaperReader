"""
LLM API客户端模块
支持OpenAI兼容的API（DeepSeek、智谱AI、通义千问等）
支持Anthropic兼容的API（GLM-4等）
"""

import time
from typing import List, Dict, Any, Optional, Iterator
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import OpenAI


class AnthropicClient:
    """
    Anthropic兼容API客户端
    支持GLM-4等使用Anthropic格式的API
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "claude-3-sonnet",
        temperature: float = 0.3,
        max_tokens: int = 4000,
        timeout: int = 60
    ):
        """初始化Anthropic客户端"""
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """发送聊天请求"""
        import requests

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        # 转换消息格式
        anthropic_messages = []
        system_message = ""

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # 构建请求 - 使用中转站支持的格式
        url = f"{self.base_url}/v1/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": tokens
        }

        if system_message:
            payload["system"] = system_message

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()
        result = response.json()

        return result["content"][0]["text"]


class LLMClient:
    """
    通用LLM客户端
    自动检测API类型并选择合适的客户端
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        max_tokens: int = 4000,
        timeout: int = 60,
        api_type: str = "auto"
    ):
        """初始化LLM客户端"""
        self.api_type = api_type
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # 自动检测API类型
        if api_type == "auto":
            if "glm" in model.lower():
                self.api_type = "anthropic"
            else:
                self.api_type = "openai"

        # 创建对应的客户端
        if self.api_type == "anthropic":
            self.client = AnthropicClient(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """发送聊天请求"""
        if self.api_type == "anthropic":
            return self.client.chat(messages, temperature, max_tokens, stream)
        else:
            temp = temperature if temperature is not None else self.temperature
            tokens = max_tokens if max_tokens is not None else self.max_tokens

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                stream=stream
            )
            return response.choices[0].message.content

    def analyze_paper(
        self,
        content: str,
        prompt: str,
        title: str = "",
        authors: str = "",
        published: str = ""
    ) -> str:
        """分析论文内容"""
        formatted_prompt = prompt.format(
            content=content,
            title=title,
            authors=authors,
            published=published,
            abstract=content[:2000] if len(content) > 2000 else content
        )

        messages = [
            {"role": "system", "content": "你是一个专业的学术研究助手，擅长分析和总结学术论文。"},
            {"role": "user", "content": formatted_prompt}
        ]

        return self.chat(messages)

    def classify_paper(self, title: str, abstract: str, prompt: str) -> str:
        """对论文进行分类"""
        formatted_prompt = prompt.format(title=title, abstract=abstract)

        messages = [
            {"role": "system", "content": "你是一个学术文献分类专家。"},
            {"role": "user", "content": formatted_prompt}
        ]

        return self.chat(messages)

    def extract_contributions(self, content: str, prompt: str) -> str:
        """提取论文创新点"""
        formatted_prompt = prompt.format(content=content)

        messages = [
            {"role": "system", "content": "你是一个学术研究助手。"},
            {"role": "user", "content": formatted_prompt}
        ]

        return self.chat(messages)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMClient":
        """从配置字典创建客户端"""
        return cls(
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url", "https://api.openai.com/v1"),
            model=config.get("model", "gpt-3.5-turbo"),
            temperature=config.get("temperature", 0.3),
            max_tokens=config.get("max_tokens", 4000),
            timeout=config.get("timeout", 60)
        )
