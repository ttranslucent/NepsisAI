"""LLM streaming API adapters for OpenAI and Anthropic.

Provides unified interface for different LLM providers.
"""

from __future__ import annotations
from typing import Iterator, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class StreamToken:
    """Unified token representation across LLM providers."""
    text: str
    index: int
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class StreamAdapter(ABC):
    """Base class for LLM streaming adapters."""

    @abstractmethod
    def stream_completion(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> Iterator[StreamToken]:
        """Stream completion tokens from LLM.

        Args:
            prompt: Input prompt
            model: Model identifier
            **kwargs: Provider-specific parameters

        Yields:
            StreamToken objects
        """
        pass


class OpenAIAdapter(StreamAdapter):
    """OpenAI streaming adapter (GPT-4, GPT-3.5, etc.)."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI adapter.

        Args:
            api_key: OpenAI API key (uses env var if None)
        """
        try:
            import openai
            self.openai = openai
            if api_key:
                self.openai.api_key = api_key
        except ImportError:
            raise ImportError(
                "OpenAI package required for streaming. "
                "Install with: pip install nepsisai[stream]"
            )

    def stream_completion(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[StreamToken]:
        """Stream completion from OpenAI.

        Args:
            prompt: Input prompt
            model: OpenAI model name
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional OpenAI parameters

        Yields:
            StreamToken objects
        """
        response = self.openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        token_idx = 0
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield StreamToken(
                    text=chunk.choices[0].delta.content,
                    index=token_idx,
                    finish_reason=chunk.choices[0].finish_reason,
                    metadata={
                        "model": model,
                        "provider": "openai"
                    }
                )
                token_idx += 1


class OllamaAdapter(StreamAdapter):
    """Ollama local model streaming adapter."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama adapter.

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.base_url = base_url.rstrip('/')

    def stream_completion(
        self,
        prompt: str,
        model: str = "llama2",
        temperature: float = 0.7,
        **kwargs
    ) -> Iterator[StreamToken]:
        """Stream completion from Ollama.

        Args:
            prompt: Input prompt
            model: Ollama model name (e.g., 'llama2', 'mistral', 'codellama')
            temperature: Sampling temperature
            **kwargs: Additional Ollama parameters

        Yields:
            StreamToken objects
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Requests package required for Ollama streaming. "
                "Install with: pip install requests"
            )

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }

        token_idx = 0
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                import json
                chunk = json.loads(line)

                if "response" in chunk and chunk["response"]:
                    yield StreamToken(
                        text=chunk["response"],
                        index=token_idx,
                        finish_reason="stop" if chunk.get("done", False) else None,
                        metadata={
                            "model": model,
                            "provider": "ollama"
                        }
                    )
                    token_idx += 1


class AnthropicAdapter(StreamAdapter):
    """Anthropic Claude streaming adapter."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Anthropic adapter.

        Args:
            api_key: Anthropic API key (uses env var if None)
        """
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Anthropic package required for streaming. "
                "Install with: pip install nepsisai[stream]"
            )

    def stream_completion(
        self,
        prompt: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Iterator[StreamToken]:
        """Stream completion from Anthropic Claude.

        Args:
            prompt: Input prompt
            model: Claude model name
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional Anthropic parameters

        Yields:
            StreamToken objects
        """
        token_idx = 0

        with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        ) as stream:
            for text in stream.text_stream:
                yield StreamToken(
                    text=text,
                    index=token_idx,
                    metadata={
                        "model": model,
                        "provider": "anthropic"
                    }
                )
                token_idx += 1


def get_adapter(model: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> StreamAdapter:
    """Get appropriate adapter for model.

    Args:
        model: Model identifier (e.g., 'gpt-4', 'claude-3-sonnet', 'llama2')
        api_key: Optional API key (for OpenAI/Anthropic)
        base_url: Optional base URL (for Ollama)

    Returns:
        StreamAdapter instance

    Raises:
        ValueError: If model not recognized
    """
    # OpenAI models
    if any(x in model.lower() for x in ['gpt', 'o1', 'davinci', 'turbo']):
        return OpenAIAdapter(api_key=api_key)

    # Anthropic models
    elif any(x in model.lower() for x in ['claude', 'sonnet', 'opus', 'haiku']):
        return AnthropicAdapter(api_key=api_key)

    # Ollama models (llama, mistral, codellama, etc.) or explicit ollama/ prefix
    elif any(x in model.lower() for x in ['llama', 'mistral', 'codellama', 'vicuna', 'orca', 'ollama/']):
        return OllamaAdapter(base_url=base_url or "http://localhost:11434")

    else:
        raise ValueError(
            f"Model '{model}' not recognized. "
            f"Supported: GPT-*, Claude-*, Llama*, Mistral*, CodeLlama*, or ollama/model-name"
        )
