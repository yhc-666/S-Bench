"""Closed source model implementations (GPT-4, DeepSeek)."""

import os
import json
import time
from typing import Dict, List, Any
import requests
from .base_model import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI GPT-4 implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config['api_key']
        self.endpoint = config['endpoint']
        self.model_name = config['model_name']
        self.timeout = config.get('timeout', 60)

    def generate_with_tags(self, prompt: str, stop_sequences: List[str] = None, **kwargs) -> str:
        """Generate response using chat completions API with stop sequences."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Put everthing in prompt (模仿raw text)
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "stop": stop_sequences
        }

        for retry in range(3):
            try:
                response = requests.post(
                    self.endpoint,  # Use the chat completions endpoint
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Get the response content
                content = response.json()['choices'][0]['message']['content']

                # Append the stop sequence that was triggered
                # The API strips stop sequences, so we need to add them back
                if stop_sequences and content:
                    # Check for each possible unclosed tag and append the appropriate closing
                    if '<search>' in content and '</search>' not in content:
                        content += '</search>'
                    elif '<answer>' in content and '</answer>' not in content:
                        content += '</answer>'

                return content
            except Exception as e:
                if retry == 2:
                    raise e
                time.sleep(2 ** retry)

    def generate_with_functions(self, messages: List[Dict[str, str]], tools: List[Dict], **kwargs) -> Dict:
        """Generate response with function/tool calling."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature)
        }

        for retry in range(3):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()['choices'][0]['message']
                return {
                    'content': result.get('content', ''),
                    'tool_calls': result.get('tool_calls', [])
                }
            except Exception as e:
                if retry == 2:
                    raise e
                time.sleep(2 ** retry)


class DeepSeekModel(BaseModel):
    """DeepSeek model implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config['api_key'] if not config['api_key'].startswith('${') else os.getenv(config['api_key'].replace('${', '').replace('}', ''))
        self.endpoint = config['endpoint']
        self.model_name = config['model_name']
        self.timeout = config.get('timeout', 60)

    def generate_with_tags(self, prompt: str, stop_sequences: List[str] = None, **kwargs) -> str:
        """Generate response using chat completions API with stop sequences."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Use chat completions endpoint with messages format
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "stop": stop_sequences
        }

        for retry in range(3):
            try:
                response = requests.post(
                    self.endpoint,  # Use the chat completions endpoint
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Get the response content
                content = response.json()['choices'][0]['message']['content']

                # Append the stop sequence that was triggered
                # The API strips stop sequences, so we need to add them back
                if stop_sequences and content:
                    # Check for each possible unclosed tag and append the appropriate closing
                    if '<search>' in content and '</search>' not in content:
                        content += '</search>'
                    elif '<answer>' in content and '</answer>' not in content:
                        content += '</answer>'

                return content
            except Exception as e:
                if retry == 2:
                    raise e
                time.sleep(2 ** retry)

    def generate_with_functions(self, messages: List[Dict[str, str]], tools: List[Dict], **kwargs) -> Dict:
        """Generate response with function/tool calling."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature)
        }

        for retry in range(3):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()['choices'][0]['message']
                return {
                    'content': result.get('content', ''),
                    'tool_calls': result.get('tool_calls', [])
                }
            except Exception as e:
                if retry == 2:
                    raise e
                time.sleep(2 ** retry)
