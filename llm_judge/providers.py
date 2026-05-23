from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from hashlib import sha256


@dataclass
class LLMResponse:
    """Provider completion response."""
    text: str
    latency_ms: int
    usage: dict[str, Any] = field(default_factory=dict)


class LLMProviderError(RuntimeError):
    """Raised when a provider call fails after retries."""
    pass


class LLMProvider:
    """Base synchronous provider interface."""
    name = "base"
    model = ""

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        """Complete a prompt.

        Parameters
        ----------
        prompt:
            Prompt text.
        json_mode:
            Request JSON-constrained output when the provider supports it.
        """
        raise NotImplementedError


class MockProvider(LLMProvider):
    """Deterministic provider for CLI smoke tests."""
    name = "mock"
    model = "mock"

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        if not json_mode:
            return LLMResponse(text="Mock answer generated from provided context.", latency_ms=0)
        if "Create a benchmark reference answer" in prompt:
            return LLMResponse(
                text=json.dumps(
                    {
                        "expected": "Mock reference answer generated from oracle context.",
                        "expected_facts": ["mock provider does not evaluate oracle context"],
                        "acceptable_answers": ["Mock reference answer"],
                        "rationale": "Mock provider returned a fixed reference for smoke testing.",
                    }
                ),
                latency_ms=0,
            )
        return LLMResponse(
            text=json.dumps(
                {
                    "verdict": "PARTIAL",
                    "score": 0.75,
                    "answer_score": 0.75,
                    "retrieval_score": 0.75,
                    "supported": ["mock provider does not evaluate semantics"],
                    "missing": [],
                    "contradictions": [],
                    "rationale": "Mock provider returned a fixed response for smoke testing.",
                }
            ),
            latency_ms=0,
        )


class CommandProvider(LLMProvider):
    """Provider that sends prompts to a shell command on stdin."""
    name = "command"

    def __init__(self, command: str, model: str = "command") -> None:
        self.command = command
        self.model = model

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                shlex.split(self.command),
                input=prompt,
                text=True,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise LLMProviderError(exc.stderr or str(exc)) from exc
        return LLMResponse(text=completed.stdout, latency_ms=int((time.perf_counter() - start) * 1000))


class HTTPProvider(LLMProvider):
    """Base class for retrying JSON-over-HTTP providers."""
    def __init__(
        self,
        name: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        retries: int = 2,
        retry_base_delay: float = 1.0,
        max_tokens: int | None = None,
        disable_response_format: bool = False,
        strict_json_fallback: bool = True,
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.retries = retries
        self.retry_base_delay = retry_base_delay
        self.max_tokens = max_tokens
        self.disable_response_format = disable_response_format
        self.strict_json_fallback = strict_json_fallback

    def _post(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> LLMResponse:
        payload = json.dumps(body).encode("utf-8")
        start = time.perf_counter()
        last_error = ""
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return LLMResponse(
                    text=self._extract_text(data),
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    usage=data.get("usage") or {},
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                last_error = f"Provider HTTP {exc.code}: {detail}"
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= self.retries:
                    break
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else self.retry_base_delay * (2**attempt)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = f"Provider transport error: {exc}"
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_base_delay * (2**attempt))
        raise LLMProviderError(last_error or "provider request failed")

    def _extract_text(self, data: dict[str, Any]) -> str:
        raise NotImplementedError


class OpenAICompatibleProvider(HTTPProvider):
    """Provider for OpenAI-compatible ``/chat/completions`` APIs."""
    def __init__(self, name: str, model: str, base_url: str, api_key: str | None, **kwargs: Any) -> None:
        super().__init__(name=name, model=model, base_url=base_url, api_key=api_key, **kwargs)

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if json_mode and not self.disable_response_format:
            body["response_format"] = {"type": "json_object"}
        try:
            return self._post(f"{self.base_url}/chat/completions", body, headers)
        except LLMProviderError:
            if not json_mode or self.disable_response_format or not self.strict_json_fallback or "response_format" not in body:
                raise
            fallback_body = dict(body)
            fallback_body.pop("response_format", None)
            return self._post(f"{self.base_url}/chat/completions", fallback_body, headers)

    def _extract_text(self, data: dict[str, Any]) -> str:
        return data["choices"][0]["message"]["content"]


class OllamaProvider(HTTPProvider):
    """Provider for Ollama ``/api/generate``."""
    def __init__(self, model: str, base_url: str = "http://localhost:11434", **kwargs: Any) -> None:
        super().__init__(name="ollama", model=model, base_url=base_url, api_key=None, **kwargs)

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        body: dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
        if self.max_tokens is not None:
            body["options"]["num_predict"] = self.max_tokens
        if json_mode and not self.disable_response_format:
            body["format"] = "json"
        try:
            return self._post(f"{self.base_url}/api/generate", body, {"Content-Type": "application/json"})
        except LLMProviderError:
            if not json_mode or self.disable_response_format or not self.strict_json_fallback or "format" not in body:
                raise
            fallback_body = dict(body)
            fallback_body.pop("format", None)
            return self._post(f"{self.base_url}/api/generate", fallback_body, {"Content-Type": "application/json"})

    def _extract_text(self, data: dict[str, Any]) -> str:
        return data.get("response", "")


class AnthropicProvider(HTTPProvider):
    """Provider for Anthropic Messages API."""
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.anthropic.com/v1", **kwargs: Any) -> None:
        super().__init__(name="anthropic", model=model, base_url=base_url, api_key=api_key, **kwargs)

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": self.model,
            "max_tokens": 1200,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        return self._post(f"{self.base_url}/messages", body, headers)

    def _extract_text(self, data: dict[str, Any]) -> str:
        parts = data.get("content") or []
        return "".join(part.get("text", "") for part in parts)


class GeminiProvider(HTTPProvider):
    """Provider for Gemini generateContent API."""
    def __init__(self, model: str, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta", **kwargs: Any) -> None:
        super().__init__(name="gemini", model=model, base_url=base_url, api_key=api_key, **kwargs)

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        if self.max_tokens is not None:
            body["generationConfig"]["maxOutputTokens"] = self.max_tokens
        if json_mode and not self.disable_response_format:
            body["generationConfig"]["response_mime_type"] = "application/json"
        try:
            return self._post(f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}", body, headers)
        except LLMProviderError:
            if not json_mode or self.disable_response_format or not self.strict_json_fallback or "response_mime_type" not in body["generationConfig"]:
                raise
            fallback_body = dict(body)
            fallback_config = dict(body["generationConfig"])
            fallback_config.pop("response_mime_type", None)
            fallback_body["generationConfig"] = fallback_config
            return self._post(f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}", fallback_body, headers)

    def _extract_text(self, data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)


class CachedProvider(LLMProvider):
    """Prompt-hash cache wrapper for any provider."""
    def __init__(self, provider: LLMProvider, cache_dir: Path, namespace: str) -> None:
        self.provider = provider
        self.cache_dir = cache_dir / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.name = provider.name
        self.model = provider.model

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        key = sha256(
            json.dumps(
                {
                    "provider": self.provider.name,
                    "model": self.provider.model,
                    "max_tokens": getattr(self.provider, "max_tokens", None),
                    "disable_response_format": getattr(self.provider, "disable_response_format", None),
                    "strict_json_fallback": getattr(self.provider, "strict_json_fallback", None),
                    "json_mode": json_mode,
                    "prompt": prompt,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return LLMResponse(text=data["text"], latency_ms=0, usage=data.get("usage") or {})
        response = self.provider.complete(prompt, json_mode=json_mode)
        path.write_text(json.dumps({"text": response.text, "usage": response.usage}, ensure_ascii=False), encoding="utf-8")
        return response


def build_provider(
    provider: str,
    model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    command: str | None,
    timeout: float,
    temperature: float,
    retries: int = 2,
    max_tokens: int | None = None,
    disable_response_format: bool = False,
    strict_json_fallback: bool = True,
) -> LLMProvider:
    """Build a configured provider from CLI/API parameters."""
    provider = provider.lower()
    api_key = os.environ.get(api_key_env or "")
    if provider == "mock":
        return MockProvider()
    if provider == "command":
        if not command:
            raise ValueError("--judge-command is required for provider=command")
        return CommandProvider(command, model=model or "command")
    if provider == "ollama":
        return OllamaProvider(
            model=model or "qwen2.5:14b",
            base_url=base_url or "http://localhost:11434",
            timeout=timeout,
            retries=retries,
            max_tokens=max_tokens,
            disable_response_format=disable_response_format,
            strict_json_fallback=strict_json_fallback,
        )
    if provider in {"openai", "openai-compatible", "openrouter"}:
        default_url = {
            "openai": "https://api.openai.com/v1",
            "openai-compatible": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }[provider]
        env = api_key_env or ("OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY")
        api_key = os.environ.get(env)
        resolved_base_url = base_url or default_url
        needs_key = provider in {"openai", "openrouter"} or "api.openai.com" in resolved_base_url
        if needs_key and not api_key:
            raise ValueError(f"Missing API key in ${env}")
        return OpenAICompatibleProvider(
            name=provider,
            model=model or "gpt-4.1-mini",
            base_url=resolved_base_url,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            retries=retries,
            max_tokens=max_tokens,
            disable_response_format=disable_response_format,
            strict_json_fallback=strict_json_fallback,
        )
    if provider == "anthropic":
        env = api_key_env or "ANTHROPIC_API_KEY"
        api_key = os.environ.get(env)
        if not api_key:
            raise ValueError(f"Missing API key in ${env}")
        return AnthropicProvider(
            model=model or "claude-3-5-sonnet-latest",
            api_key=api_key,
            base_url=base_url or "https://api.anthropic.com/v1",
            timeout=timeout,
            retries=retries,
            max_tokens=max_tokens,
            disable_response_format=disable_response_format,
            strict_json_fallback=strict_json_fallback,
        )
    if provider == "gemini":
        env = api_key_env or "GEMINI_API_KEY"
        api_key = os.environ.get(env)
        if not api_key:
            raise ValueError(f"Missing API key in ${env}")
        return GeminiProvider(
            model=model or "gemini-2.5-flash",
            api_key=api_key,
            base_url=base_url or "https://generativelanguage.googleapis.com/v1beta",
            timeout=timeout,
            retries=retries,
            max_tokens=max_tokens,
            disable_response_format=disable_response_format,
            strict_json_fallback=strict_json_fallback,
        )
    raise ValueError(f"Unsupported provider: {provider}")
