"""
Model Backends

Abstract backend interface and concrete implementations for various model providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from distillo.config import ModelConfig


class ModelBackend(ABC):
    """Abstract base class for model backends"""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate(
        self, prompts: List[str], generation_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate completions for a batch of prompts

        Args:
            prompts: List of input prompts
            generation_params: Generation parameters (temperature, top_p, etc.)

        Returns:
            List of generation results with text and metadata
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up resources"""
        pass


class VLLMBackend(ModelBackend):
    """vLLM backend for local model inference"""

    def __init__(self, config: ModelConfig):
        if not VLLM_AVAILABLE:
            raise ImportError("vllm is required. Install with: pip install distillo[server]")

        super().__init__(config)
        self._llm = None
        self._initialize()

    def _initialize(self):
        """Initialize vLLM model"""
        # Extract vLLM-specific parameters
        vllm_kwargs = {
            "model": self.config.name,
            "trust_remote_code": True,
        }

        # Add optional parameters from config
        if "tensor_parallel_size" in self.config.parameters:
            vllm_kwargs["tensor_parallel_size"] = self.config.parameters["tensor_parallel_size"]

        if "gpu_memory_utilization" in self.config.parameters:
            vllm_kwargs["gpu_memory_utilization"] = self.config.parameters[
                "gpu_memory_utilization"
            ]

        if "max_model_len" in self.config.parameters:
            vllm_kwargs["max_model_len"] = self.config.parameters["max_model_len"]

        if "dtype" in self.config.parameters:
            vllm_kwargs["dtype"] = self.config.parameters["dtype"]

        # Add any other parameters
        for key, value in self.config.parameters.items():
            if key not in vllm_kwargs:
                vllm_kwargs[key] = value

        self._llm = LLM(**vllm_kwargs)

    def generate(
        self, prompts: List[str], generation_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate completions using vLLM

        Args:
            prompts: List of input prompts
            generation_params: Generation parameters

        Returns:
            List of generation results
        """
        # Build sampling params
        sampling_kwargs = {}

        if "temperature" in generation_params:
            sampling_kwargs["temperature"] = generation_params["temperature"]

        if "top_p" in generation_params:
            sampling_kwargs["top_p"] = generation_params["top_p"]

        if "top_k" in generation_params:
            sampling_kwargs["top_k"] = generation_params["top_k"]

        if "max_tokens" in generation_params:
            sampling_kwargs["max_tokens"] = generation_params["max_tokens"]

        if "presence_penalty" in generation_params:
            sampling_kwargs["presence_penalty"] = generation_params["presence_penalty"]

        if "frequency_penalty" in generation_params:
            sampling_kwargs["frequency_penalty"] = generation_params["frequency_penalty"]

        if "stop" in generation_params:
            sampling_kwargs["stop"] = generation_params["stop"]

        if "seed" in generation_params:
            sampling_kwargs["seed"] = generation_params["seed"]

        sampling_params = SamplingParams(**sampling_kwargs)

        # Generate
        outputs = self._llm.generate(prompts, sampling_params)

        # Format results
        results = []
        for output in outputs:
            result = {
                "prompt": output.prompt,
                "text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
                "tokens": len(output.outputs[0].token_ids),
            }
            results.append(result)

        return results

    def close(self):
        """Clean up vLLM resources"""
        if self._llm is not None:
            del self._llm
            self._llm = None


class OpenAIBackend(ModelBackend):
    """OpenAI API backend"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import openai

            self.openai = openai
        except ImportError:
            raise ImportError("openai is required for OpenAI backend")

        # Get API key from parameters
        api_key = self.config.parameters.get("api_key")
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = openai.OpenAI()  # Uses OPENAI_API_KEY env var

    def generate(
        self, prompts: List[str], generation_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate completions using OpenAI API"""
        results = []

        for prompt in prompts:
            # Build request parameters
            request_params = {
                "model": self.config.name,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Add generation parameters
            if "temperature" in generation_params:
                request_params["temperature"] = generation_params["temperature"]
            if "top_p" in generation_params:
                request_params["top_p"] = generation_params["top_p"]
            if "max_tokens" in generation_params:
                request_params["max_tokens"] = generation_params["max_tokens"]
            if "presence_penalty" in generation_params:
                request_params["presence_penalty"] = generation_params["presence_penalty"]
            if "frequency_penalty" in generation_params:
                request_params["frequency_penalty"] = generation_params["frequency_penalty"]
            if "stop" in generation_params:
                request_params["stop"] = generation_params["stop"]

            # Make request
            response = self.client.chat.completions.create(**request_params)

            # Format result
            result = {
                "prompt": prompt,
                "text": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "tokens": response.usage.completion_tokens,
            }
            results.append(result)

        return results

    def close(self):
        """Clean up OpenAI resources"""
        pass


class DeepSeekBackend(ModelBackend):
    """DeepSeek API backend (OpenAI-compatible)"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import openai

            self.openai = openai
        except ImportError:
            raise ImportError("openai is required for DeepSeek backend")

        # Get API key from parameters
        api_key = self.config.parameters.get("api_key")

        # DeepSeek uses OpenAI-compatible API with custom base URL
        if api_key:
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
        else:
            # Uses DEEPSEEK_API_KEY env var
            import os
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY or provide api_key in config")
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

    def generate(
        self, prompts: List[str], generation_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate completions using DeepSeek API"""
        results = []

        for prompt in prompts:
            # Build request parameters
            request_params = {
                "model": self.config.name,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Add generation parameters
            if "temperature" in generation_params:
                request_params["temperature"] = generation_params["temperature"]
            if "top_p" in generation_params:
                request_params["top_p"] = generation_params["top_p"]
            if "max_tokens" in generation_params:
                request_params["max_tokens"] = generation_params["max_tokens"]
            if "presence_penalty" in generation_params:
                request_params["presence_penalty"] = generation_params["presence_penalty"]
            if "frequency_penalty" in generation_params:
                request_params["frequency_penalty"] = generation_params["frequency_penalty"]
            if "stop" in generation_params:
                request_params["stop"] = generation_params["stop"]

            # Make request
            response = self.client.chat.completions.create(**request_params)

            # Format result
            result = {
                "prompt": prompt,
                "text": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "tokens": response.usage.completion_tokens,
            }
            results.append(result)

        return results

    def close(self):
        """Clean up DeepSeek resources"""
        pass


class ZAIBackend(ModelBackend):
    """Z.AI (GLM) API backend (OpenAI-compatible)"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import openai

            self.openai = openai
        except ImportError:
            raise ImportError("openai is required for Z.AI backend")

        # Get API key from parameters
        api_key = self.config.parameters.get("api_key")

        # Z.AI uses OpenAI-compatible API with custom base URL
        if api_key:
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.z.ai/api/paas/v4/"
            )
        else:
            # Uses ZAI_API_KEY env var
            import os
            api_key = os.getenv("ZAI_API_KEY")
            if not api_key:
                raise ValueError("Z.AI API key not found. Set ZAI_API_KEY or provide api_key in config")
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.z.ai/api/paas/v4/"
            )

    def generate(
        self, prompts: List[str], generation_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate completions using Z.AI API"""
        results = []

        for prompt in prompts:
            # Build request parameters
            request_params = {
                "model": self.config.name,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Add generation parameters
            if "temperature" in generation_params:
                request_params["temperature"] = generation_params["temperature"]
            if "top_p" in generation_params:
                request_params["top_p"] = generation_params["top_p"]
            if "max_tokens" in generation_params:
                request_params["max_tokens"] = generation_params["max_tokens"]
            if "presence_penalty" in generation_params:
                request_params["presence_penalty"] = generation_params["presence_penalty"]
            if "frequency_penalty" in generation_params:
                request_params["frequency_penalty"] = generation_params["frequency_penalty"]
            if "stop" in generation_params:
                request_params["stop"] = generation_params["stop"]

            # Make request
            response = self.client.chat.completions.create(**request_params)

            # Format result
            result = {
                "prompt": prompt,
                "text": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "tokens": response.usage.completion_tokens,
            }
            results.append(result)

        return results

    def close(self):
        """Clean up Z.AI resources"""
        pass


class AnthropicBackend(ModelBackend):
    """Anthropic API backend"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import anthropic

            self.anthropic = anthropic
        except ImportError:
            raise ImportError("anthropic is required for Anthropic backend")

        # Get API key from parameters
        api_key = self.config.parameters.get("api_key")
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    def generate(
        self, prompts: List[str], generation_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate completions using Anthropic API"""
        results = []

        for prompt in prompts:
            # Build request parameters
            request_params = {
                "model": self.config.name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": generation_params.get("max_tokens", 4096),
            }

            # Add generation parameters
            if "temperature" in generation_params:
                request_params["temperature"] = generation_params["temperature"]
            if "top_p" in generation_params:
                request_params["top_p"] = generation_params["top_p"]
            if "top_k" in generation_params:
                request_params["top_k"] = generation_params["top_k"]
            if "stop_sequences" in generation_params:
                request_params["stop_sequences"] = generation_params["stop_sequences"]

            # Make request
            response = self.client.messages.create(**request_params)

            # Format result
            result = {
                "prompt": prompt,
                "text": response.content[0].text,
                "finish_reason": response.stop_reason,
                "tokens": response.usage.output_tokens,
            }
            results.append(result)

        return results

    def close(self):
        """Clean up Anthropic resources"""
        pass


def create_backend(config: ModelConfig) -> ModelBackend:
    """
    Factory function to create appropriate backend

    Args:
        config: Model configuration

    Returns:
        ModelBackend instance

    Raises:
        ValueError: If provider is not supported
    """
    provider = config.provider.lower()

    if provider == "vllm":
        return VLLMBackend(config)
    elif provider == "openai":
        return OpenAIBackend(config)
    elif provider == "deepseek":
        return DeepSeekBackend(config)
    elif provider == "zai" or provider == "glm":
        return ZAIBackend(config)
    elif provider == "anthropic":
        return AnthropicBackend(config)
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: vllm, openai, deepseek, zai, anthropic"
        )
