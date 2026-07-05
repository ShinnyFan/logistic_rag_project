import json
import re
from typing import List, Dict, Optional
from openai import OpenAI


class QwenChain:
    """
    Qwen API 调用封装，兼容 OpenAI Chat 接口。
    支持普通生成和流式输出。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-max",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate(self, messages: List[Dict[str, str]], stream: bool = False) -> str:
        if stream:
            return self._generate_stream(messages)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Qwen API 调用失败: {e}")
            return f"[API调用失败: {str(e)}]"

    def _generate_stream(self, messages: List[Dict[str, str]]) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            full_response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            print()
            return full_response
        except Exception as e:
            print(f"Qwen API 流式调用失败: {e}")
            return f"[API调用失败: {str(e)}]"

    def generate_direct(self, question: str) -> str:
        from .prompt_templates import DIRECT_QWEN_PROMPT_TEMPLATE
        prompt = DIRECT_QWEN_PROMPT_TEMPLATE.format(question=question)
        messages = [
            {"role": "system", "content": "你是一个专业的物流调度算法专家。"},
            {"role": "user", "content": prompt},
        ]
        return self.generate(messages)


def parse_judge_response(response: str) -> Dict:
    try:
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(response)
    except (json.JSONDecodeError, AttributeError):
        return {
            "relevance": 0,
            "consistency": 0,
            "completeness": 0,
            "overall": 0.0,
            "reason": f"JSON解析失败，原始响应: {response[:200]}",
        }