"""
API.py - A Complete Single-File Library for OpenAI-Standard /chat/completions API Inference

This library mirrors the OpenAI Python SDK interface but routes everything through
the /chat/completions endpoint. Supports chat, tools/functions, vision (images),
streaming, JSON mode, embeddings simulation, and more.

Usage:
    from API import Client
    
    client = Client(api_key="sk-...", base_url="http://localhost:8000/v1")
    response = client.chat.completions.create(
        model="my-model",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)
"""

from __future__ import annotations

import json
import time
import uuid
import base64
import struct
import hashlib
import os
import re
import io
import copy
import mimetypes
import urllib.request
import urllib.error
import urllib.parse
import ssl
import threading
import queue
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)
from enum import Enum

# =============================================================================
# Exceptions
# =============================================================================

class APIError(Exception):
    """Base exception for API errors."""
    def __init__(self, message: str, status_code: int = None, response: Any = None, body: Any = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response
        self.body = body

    def __str__(self):
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthenticationError(APIError):
    """Raised for 401 errors."""
    pass


class PermissionDeniedError(APIError):
    """Raised for 403 errors."""
    pass


class NotFoundError(APIError):
    """Raised for 404 errors."""
    pass


class RateLimitError(APIError):
    """Raised for 429 errors."""
    pass


class BadRequestError(APIError):
    """Raised for 400 errors."""
    pass


class InternalServerError(APIError):
    """Raised for 500+ errors."""
    pass


class APIConnectionError(APIError):
    """Raised when connection fails."""
    pass


class APITimeoutError(APIConnectionError):
    """Raised on timeout."""
    pass


class ContentFilterError(APIError):
    """Raised when content is filtered."""
    pass


# =============================================================================
# Data Models
# =============================================================================

class BaseModel:
    """Base model with dict-like access and serialization."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if isinstance(value, BaseModel):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, BaseModel) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def model_dump(self, exclude_none: bool = True) -> dict:
        d = self.to_dict()
        if exclude_none:
            return {k: v for k, v in d.items() if v is not None}
        return d

    def model_dump_json(self, indent: int = None) -> str:
        return json.dumps(self.model_dump(), indent=indent, default=str)

    def __repr__(self):
        fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if v is not None)
        return f"{self.__class__.__name__}({fields})"

    def __str__(self):
        return self.__repr__()


class FunctionCall(BaseModel):
    """Represents a function call."""
    def __init__(self, name: str = None, arguments: str = None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.arguments = arguments


class ToolCall(BaseModel):
    """Represents a tool call."""
    def __init__(self, id: str = None, type: str = "function", function: FunctionCall = None, **kwargs):
        super().__init__(**kwargs)
        self.id = id or f"call_{uuid.uuid4().hex[:24]}"
        self.type = type
        self.function = function


class TopLogprob(BaseModel):
    """Top logprob entry."""
    def __init__(self, token: str = None, logprob: float = None, bytes: list = None, **kwargs):
        super().__init__(**kwargs)
        self.token = token
        self.logprob = logprob
        self.bytes = bytes


class TokenLogprob(BaseModel):
    """Token logprob entry."""
    def __init__(self, token: str = None, logprob: float = None, bytes: list = None, top_logprobs: List[TopLogprob] = None, **kwargs):
        super().__init__(**kwargs)
        self.token = token
        self.logprob = logprob
        self.bytes = bytes
        self.top_logprobs = top_logprobs


class ChoiceLogprobs(BaseModel):
    """Logprobs for a choice."""
    def __init__(self, content: List[TokenLogprob] = None, **kwargs):
        super().__init__(**kwargs)
        self.content = content


class ChatMessage(BaseModel):
    """Represents a chat message."""
    def __init__(
        self,
        role: str = None,
        content: Union[str, list, None] = None,
        name: str = None,
        function_call: FunctionCall = None,
        tool_calls: List[ToolCall] = None,
        tool_call_id: str = None,
        refusal: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.name = name
        self.function_call = function_call
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.refusal = refusal


class DeltaMessage(BaseModel):
    """Represents a streaming delta message."""
    def __init__(
        self,
        role: str = None,
        content: str = None,
        function_call: FunctionCall = None,
        tool_calls: List[ToolCall] = None,
        refusal: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.function_call = function_call
        self.tool_calls = tool_calls
        self.refusal = refusal


class Choice(BaseModel):
    """Represents a completion choice."""
    def __init__(
        self,
        index: int = 0,
        message: ChatMessage = None,
        finish_reason: str = None,
        logprobs: ChoiceLogprobs = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index = index
        self.message = message
        self.finish_reason = finish_reason
        self.logprobs = logprobs


class StreamChoice(BaseModel):
    """Represents a streaming choice."""
    def __init__(
        self,
        index: int = 0,
        delta: DeltaMessage = None,
        finish_reason: str = None,
        logprobs: ChoiceLogprobs = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index = index
        self.delta = delta
        self.finish_reason = finish_reason
        self.logprobs = logprobs


class Usage(BaseModel):
    """Token usage information."""
    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class ChatCompletion(BaseModel):
    """Represents a chat completion response."""
    def __init__(
        self,
        id: str = None,
        object: str = "chat.completion",
        created: int = None,
        model: str = None,
        choices: List[Choice] = None,
        usage: Usage = None,
        system_fingerprint: str = None,
        service_tier: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.id = id or f"chatcmpl-{uuid.uuid4().hex[:29]}"
        self.object = object
        self.created = created or int(time.time())
        self.model = model
        self.choices = choices or []
        self.usage = usage
        self.system_fingerprint = system_fingerprint
        self.service_tier = service_tier


class ChatCompletionChunk(BaseModel):
    """Represents a streaming chat completion chunk."""
    def __init__(
        self,
        id: str = None,
        object: str = "chat.completion.chunk",
        created: int = None,
        model: str = None,
        choices: List[StreamChoice] = None,
        usage: Usage = None,
        system_fingerprint: str = None,
        service_tier: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.id = id or f"chatcmpl-{uuid.uuid4().hex[:29]}"
        self.object = object
        self.created = created or int(time.time())
        self.model = model
        self.choices = choices or []
        self.usage = usage
        self.system_fingerprint = system_fingerprint
        self.service_tier = service_tier


class EmbeddingData(BaseModel):
    """Single embedding result."""
    def __init__(self, object: str = "embedding", embedding: list = None, index: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.object = object
        self.embedding = embedding or []
        self.index = index


class EmbeddingResponse(BaseModel):
    """Embeddings response."""
    def __init__(self, object: str = "list", data: list = None, model: str = None, usage: Usage = None, **kwargs):
        super().__init__(**kwargs)
        self.object = object
        self.data = data or []
        self.model = model
        self.usage = usage


class ImageData(BaseModel):
    """Single image result."""
    def __init__(self, url: str = None, b64_json: str = None, revised_prompt: str = None, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self.b64_json = b64_json
        self.revised_prompt = revised_prompt


class ImageResponse(BaseModel):
    """Image generation response."""
    def __init__(self, created: int = None, data: List[ImageData] = None, **kwargs):
        super().__init__(**kwargs)
        self.created = created or int(time.time())
        self.data = data or []


class AudioTranscription(BaseModel):
    """Audio transcription result."""
    def __init__(self, text: str = None, language: str = None, duration: float = None, segments: list = None, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.language = language
        self.duration = duration
        self.segments = segments


class AudioTranslation(BaseModel):
    """Audio translation result."""
    def __init__(self, text: str = None, **kwargs):
        super().__init__(**kwargs)
        self.text = text


class AudioSpeechResponse(BaseModel):
    """Text-to-speech response."""
    def __init__(self, content: bytes = None, **kwargs):
        super().__init__(**kwargs)
        self.content = content

    def stream_to_file(self, filepath: str):
        with open(filepath, "wb") as f:
            f.write(self.content)

    def iter_bytes(self, chunk_size: int = 4096) -> Iterator[bytes]:
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]


class ModerationCategory(BaseModel):
    """Moderation category result."""
    pass


class ModerationResult(BaseModel):
    """Single moderation result."""
    def __init__(self, flagged: bool = False, categories: dict = None, category_scores: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.flagged = flagged
        self.categories = categories or {}
        self.category_scores = category_scores or {}


class ModerationResponse(BaseModel):
    """Moderation response."""
    def __init__(self, id: str = None, model: str = None, results: List[ModerationResult] = None, **kwargs):
        super().__init__(**kwargs)
        self.id = id or f"modr-{uuid.uuid4().hex[:29]}"
        self.model = model
        self.results = results or []


class ModelInfo(BaseModel):
    """Model information."""
    def __init__(self, id: str = None, object: str = "model", created: int = None, owned_by: str = None, **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self.object = object
        self.created = created or int(time.time())
        self.owned_by = owned_by or "organization"


class ModelList(BaseModel):
    """List of models."""
    def __init__(self, object: str = "list", data: List[ModelInfo] = None, **kwargs):
        super().__init__(**kwargs)
        self.object = object
        self.data = data or []


# =============================================================================
# Stream Wrapper
# =============================================================================

class Stream:
    """Wrapper for streaming responses that supports iteration."""

    def __init__(self, response_iterator: Iterator[ChatCompletionChunk]):
        self._iterator = response_iterator
        self._collected_chunks: List[ChatCompletionChunk] = []
        self._consumed = False

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self._stream()

    def __next__(self) -> ChatCompletionChunk:
        return next(self._stream())

    def _stream(self) -> Iterator[ChatCompletionChunk]:
        for chunk in self._iterator:
            self._collected_chunks.append(chunk)
            yield chunk
        self._consumed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Consume remaining stream."""
        if not self._consumed:
            for _ in self._stream():
                pass

    def collect(self) -> ChatCompletion:
        """Consume the entire stream and return a merged ChatCompletion."""
        if not self._consumed:
            for _ in self._stream():
                pass

        if not self._collected_chunks:
            return ChatCompletion()

        # Merge chunks into a single completion
        merged_content = {}  # index -> content parts
        merged_tool_calls = {}  # index -> {tc_index -> ToolCall}
        merged_function_call = {}  # index -> FunctionCall
        finish_reasons = {}
        model = None
        completion_id = None
        created = None
        roles = {}

        for chunk in self._collected_chunks:
            if chunk.model:
                model = chunk.model
            if chunk.id:
                completion_id = chunk.id
            if chunk.created:
                created = chunk.created

            for choice in (chunk.choices or []):
                idx = choice.index
                delta = choice.delta
                if not delta:
                    continue

                if delta.role:
                    roles[idx] = delta.role

                if delta.content:
                    merged_content.setdefault(idx, [])
                    merged_content[idx].append(delta.content)

                if delta.tool_calls:
                    merged_tool_calls.setdefault(idx, {})
                    for tc in delta.tool_calls:
                        tc_idx = tc.index if hasattr(tc, 'index') and tc.index is not None else 0
                        if tc_idx not in merged_tool_calls[idx]:
                            merged_tool_calls[idx][tc_idx] = ToolCall(
                                id=tc.id or f"call_{uuid.uuid4().hex[:24]}",
                                type=tc.type or "function",
                                function=FunctionCall(name="", arguments="")
                            )
                        existing = merged_tool_calls[idx][tc_idx]
                        if tc.id:
                            existing.id = tc.id
                        if tc.function:
                            if tc.function.name:
                                existing.function.name += tc.function.name
                            if tc.function.arguments:
                                existing.function.arguments += tc.function.arguments

                if delta.function_call:
                    if idx not in merged_function_call:
                        merged_function_call[idx] = FunctionCall(name="", arguments="")
                    if delta.function_call.name:
                        merged_function_call[idx].name += delta.function_call.name
                    if delta.function_call.arguments:
                        merged_function_call[idx].arguments += delta.function_call.arguments

                if choice.finish_reason:
                    finish_reasons[idx] = choice.finish_reason

        # Build choices
        all_indices = set(list(merged_content.keys()) + list(merged_tool_calls.keys()) +
                        list(merged_function_call.keys()) + list(finish_reasons.keys()) +
                        list(roles.keys()))

        choices = []
        for idx in sorted(all_indices):
            content = "".join(merged_content.get(idx, [])) or None
            tool_calls_list = None
            if idx in merged_tool_calls:
                tool_calls_list = [merged_tool_calls[idx][k] for k in sorted(merged_tool_calls[idx].keys())]
            func_call = merged_function_call.get(idx)

            msg = ChatMessage(
                role=roles.get(idx, "assistant"),
                content=content,
                tool_calls=tool_calls_list,
                function_call=func_call,
            )
            choices.append(Choice(
                index=idx,
                message=msg,
                finish_reason=finish_reasons.get(idx, "stop"),
            ))

        return ChatCompletion(
            id=completion_id,
            object="chat.completion",
            created=created,
            model=model,
            choices=choices,
            usage=self._collected_chunks[-1].usage if self._collected_chunks else None,
        )

    @property
    def response(self):
        return self


# =============================================================================
# HTTP Transport
# =============================================================================

class HTTPTransport:
    """Low-level HTTP transport using urllib."""

    def __init__(
        self,
        base_url: str,
        api_key: str = None,
        organization: str = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        default_headers: dict = None,
        verify_ssl: bool = True,
        proxy: str = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.organization = organization
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_headers = default_headers or {}
        self.verify_ssl = verify_ssl
        self.proxy = proxy

        if not self.verify_ssl:
            self._ssl_context = ssl.create_default_context()
            self._ssl_context.check_hostname = False
            self._ssl_context.verify_mode = ssl.CERT_NONE
        else:
            self._ssl_context = ssl.create_default_context()

    def _build_headers(self, extra_headers: dict = None, content_type: str = "application/json") -> dict:
        headers = {
            "User-Agent": "API.py/1.0",
            "Accept": "application/json",
        }
        if content_type:
            headers["Content-Type"] = content_type
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        headers.update(self.default_headers)
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _handle_error(self, status_code: int, body: str, url: str):
        """Raise appropriate exception based on status code."""
        try:
            error_body = json.loads(body)
            message = error_body.get("error", {}).get("message", body) if isinstance(error_body.get("error"), dict) else str(error_body.get("error", body))
        except (json.JSONDecodeError, AttributeError):
            message = body or f"HTTP {status_code}"

        error_map = {
            400: BadRequestError,
            401: AuthenticationError,
            403: PermissionDeniedError,
            404: NotFoundError,
            429: RateLimitError,
        }

        error_class = error_map.get(status_code, InternalServerError if status_code >= 500 else APIError)
        raise error_class(message=message, status_code=status_code, body=body)

    def request(
        self,
        method: str,
        path: str,
        body: dict = None,
        headers: dict = None,
        stream: bool = False,
        content_type: str = "application/json",
        raw_body: bytes = None,
    ) -> Union[dict, Iterator[str]]:
        """Make an HTTP request."""
        url = f"{self.base_url}{path}"
        req_headers = self._build_headers(headers, content_type=content_type)

        if raw_body is not None:
            data = raw_body
        elif body is not None:
            data = json.dumps(body, default=str).encode("utf-8")
        else:
            data = None

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url=url,
                    data=data,
                    headers=req_headers,
                    method=method.upper(),
                )

                if self.proxy:
                    proxy_handler = urllib.request.ProxyHandler({
                        "http": self.proxy,
                        "https": self.proxy,
                    })
                    opener = urllib.request.build_opener(
                        proxy_handler,
                        urllib.request.HTTPSHandler(context=self._ssl_context)
                    )
                else:
                    opener = urllib.request.build_opener(
                        urllib.request.HTTPSHandler(context=self._ssl_context)
                    )

                response = opener.open(req, timeout=self.timeout)

                if stream:
                    return self._stream_response(response)

                response_body = response.read().decode("utf-8")
                try:
                    return json.loads(response_body)
                except json.JSONDecodeError:
                    return {"raw": response_body}

            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8") if e.fp else ""
                if e.code == 429 and attempt < self.max_retries:
                    retry_after = e.headers.get("Retry-After", str(2 ** attempt))
                    try:
                        wait_time = float(retry_after)
                    except ValueError:
                        wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    last_error = e
                    continue
                if e.code >= 500 and attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    last_error = e
                    continue
                self._handle_error(e.code, error_body, url)

            except urllib.error.URLError as e:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    last_error = e
                    continue
                raise APIConnectionError(
                    message=f"Connection error: {str(e.reason)}",
                    status_code=None,
                )

            except TimeoutError:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    last_error = e
                    continue
                raise APITimeoutError(
                    message=f"Request timed out after {self.timeout}s",
                    status_code=None,
                )

        raise APIConnectionError(
            message=f"Max retries ({self.max_retries}) exceeded. Last error: {last_error}",
            status_code=None,
        )

    def _stream_response(self, response) -> Iterator[str]:
        """Read SSE stream line by line."""
        buffer = ""
        while True:
            chunk = response.read(1)
            if not chunk:
                if buffer.strip():
                    yield buffer
                break
            char = chunk.decode("utf-8", errors="replace")
            buffer += char
            if char == "\n":
                line = buffer.strip()
                if line:
                    yield line
                buffer = ""

    def request_raw(
        self,
        method: str,
        path: str,
        body: dict = None,
        headers: dict = None,
        content_type: str = "application/json",
    ) -> bytes:
        """Make request and return raw bytes."""
        url = f"{self.base_url}{path}"
        req_headers = self._build_headers(headers, content_type=content_type)

        if body is not None:
            data = json.dumps(body, default=str).encode("utf-8")
        else:
            data = None

        try:
            req = urllib.request.Request(url=url, data=data, headers=req_headers, method=method.upper())
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=self._ssl_context)
            )
            response = opener.open(req, timeout=self.timeout)
            return response.read()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            self._handle_error(e.code, error_body, url)
        except urllib.error.URLError as e:
            raise APIConnectionError(message=f"Connection error: {str(e.reason)}")

    def multipart_request(
        self,
        method: str,
        path: str,
        fields: dict,
        files: dict = None,
        headers: dict = None,
    ) -> dict:
        """Make a multipart/form-data request."""
        boundary = f"----FormBoundary{uuid.uuid4().hex}"
        body_parts = []

        for key, value in fields.items():
            if value is None:
                continue
            body_parts.append(f"--{boundary}\r\n".encode())
            body_parts.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
            body_parts.append(f"{value}\r\n".encode())

        if files:
            for key, file_info in files.items():
                if file_info is None:
                    continue
                if isinstance(file_info, tuple):
                    filename, file_data = file_info[0], file_info[1]
                    content_type = file_info[2] if len(file_info) > 2 else "application/octet-stream"
                elif isinstance(file_info, (bytes, bytearray)):
                    filename = key
                    file_data = file_info
                    content_type = "application/octet-stream"
                elif isinstance(file_info, str):
                    filename = os.path.basename(file_info)
                    with open(file_info, "rb") as f:
                        file_data = f.read()
                    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                else:
                    # File-like object
                    filename = getattr(file_info, 'name', key)
                    if isinstance(filename, str) and os.sep in filename:
                        filename = os.path.basename(filename)
                    file_data = file_info.read()
                    content_type = "application/octet-stream"

                body_parts.append(f"--{boundary}\r\n".encode())
                body_parts.append(
                    f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode()
                )
                body_parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
                body_parts.append(file_data if isinstance(file_data, bytes) else file_data.encode())
                body_parts.append(b"\r\n")

        body_parts.append(f"--{boundary}--\r\n".encode())
        raw_body = b"".join(body_parts)

        req_headers = self._build_headers(headers, content_type=None)
        req_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

        url = f"{self.base_url}{path}"
        try:
            req = urllib.request.Request(url=url, data=raw_body, headers=req_headers, method=method.upper())
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=self._ssl_context)
            )
            response = opener.open(req, timeout=self.timeout)
            response_body = response.read().decode("utf-8")
            try:
                return json.loads(response_body)
            except json.JSONDecodeError:
                return {"raw": response_body, "text": response_body}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            self._handle_error(e.code, error_body, url)
        except urllib.error.URLError as e:
            raise APIConnectionError(message=f"Connection error: {str(e.reason)}")


# =============================================================================
# Message Builder Helpers
# =============================================================================

class MessageBuilder:
    """Helper to build messages with various content types."""

    @staticmethod
    def text(role: str, content: str, **kwargs) -> dict:
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        return msg

    @staticmethod
    def system(content: str, **kwargs) -> dict:
        return MessageBuilder.text("system", content, **kwargs)

    @staticmethod
    def user(content: Union[str, list], **kwargs) -> dict:
        msg = {"role": "user", "content": content}
        msg.update(kwargs)
        return msg

    @staticmethod
    def assistant(content: str = None, tool_calls: list = None, function_call: dict = None, **kwargs) -> dict:
        msg = {"role": "assistant"}
        if content is not None:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if function_call:
            msg["function_call"] = function_call
        msg.update(kwargs)
        return msg

    @staticmethod
    def tool(content: str, tool_call_id: str, **kwargs) -> dict:
        msg = {"role": "tool", "content": content, "tool_call_id": tool_call_id}
        msg.update(kwargs)
        return msg

    @staticmethod
    def function_result(name: str, content: str, **kwargs) -> dict:
        msg = {"role": "function", "name": name, "content": content}
        msg.update(kwargs)
        return msg

    @staticmethod
    def image_url(text: str, image_url: str, detail: str = "auto") -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url, "detail": detail}},
            ],
        }

    @staticmethod
    def image_base64(text: str, base64_data: str, media_type: str = "image/png", detail: str = "auto") -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_data}",
                        "detail": detail,
                    },
                },
            ],
        }

    @staticmethod
    def image_file(text: str, file_path: str, detail: str = "auto") -> dict:
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        media_type = mimetypes.guess_type(file_path)[0] or "image/png"
        return MessageBuilder.image_base64(text, data, media_type, detail)

    @staticmethod
    def multi_image(text: str, image_urls: List[str], detail: str = "auto") -> dict:
        content = [{"type": "text", "text": text}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
        return {"role": "user", "content": content}

    @staticmethod
    def audio_input(text: str, audio_base64: str, audio_format: str = "wav") -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": audio_format,
                    },
                },
            ],
        }


# =============================================================================
# Tool / Function Definition Helpers
# =============================================================================

class ToolDefinition:
    """Helper to define tools/functions for the API."""

    @staticmethod
    def function(
        name: str,
        description: str = "",
        parameters: dict = None,
        strict: bool = None,
    ) -> dict:
        func_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters or {"type": "object", "properties": {}, "required": []},
            },
        }
        if strict is not None:
            func_def["function"]["strict"] = strict
        return func_def

    @staticmethod
    def function_from_callable(func: Callable, description: str = None) -> dict:
        """Create a tool definition from a Python function using its signature."""
        import inspect
        sig = inspect.signature(func)
        properties = {}
        required = []

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            annotation = param.annotation
            if annotation == inspect.Parameter.empty:
                param_type = "string"
            else:
                param_type = type_map.get(annotation, "string")

            properties[param_name] = {"type": param_type}

            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            else:
                properties[param_name]["default"] = param.default

        func_description = description or func.__doc__ or f"Function: {func.__name__}"

        return ToolDefinition.function(
            name=func.__name__,
            description=func_description.strip(),
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )

    @staticmethod
    def response_format_json(name: str = "json_response", schema: dict = None, strict: bool = True) -> dict:
        """Create a JSON response format specification."""
        result = {"type": "json_schema"}
        json_schema = {"name": name, "strict": strict}
        if schema:
            json_schema["schema"] = schema
        result["json_schema"] = json_schema
        return result

    @staticmethod
    def response_format_text() -> dict:
        return {"type": "text"}

    @staticmethod
    def response_format_json_object() -> dict:
        return {"type": "json_object"}


# =============================================================================
# Tool Executor
# =============================================================================

class ToolExecutor:
    """Manages tool/function registration and automatic execution loops."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_definitions: List[dict] = []

    def register(self, func: Callable = None, *, name: str = None, description: str = None):
        """Register a function as a tool. Can be used as a decorator."""
        def decorator(f):
            tool_name = name or f.__name__
            self._tools[tool_name] = f
            tool_def = ToolDefinition.function_from_callable(f, description)
            if name:
                tool_def["function"]["name"] = tool_name
            self._tool_definitions.append(tool_def)
            return f

        if func is not None:
            return decorator(func)
        return decorator

    @property
    def definitions(self) -> List[dict]:
        return self._tool_definitions

    def execute(self, tool_name: str, arguments: Union[str, dict]) -> str:
        """Execute a registered tool."""
        if tool_name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return json.dumps({"error": f"Invalid JSON arguments: {arguments}"})

        try:
            result = self._tools[tool_name](**arguments)
            if isinstance(result, str):
                return result
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[dict]:
        """Execute multiple tool calls and return tool messages."""
        messages = []
        for tc in tool_calls:
            result = self.execute(tc.function.name, tc.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        return messages

    def run_conversation(
        self,
        client,
        model: str,
        messages: List[dict],
        max_iterations: int = 10,
        **kwargs,
    ) -> ChatCompletion:
        """Run a full tool-use conversation loop until completion."""
        messages = list(messages)
        kwargs.pop("tools", None)
        kwargs.pop("stream", None)

        for _ in range(max_iterations):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=self._tool_definitions,
                stream=False,
                **kwargs,
            )

            choice = response.choices[0]
            assistant_msg = choice.message

            # Build assistant message dict
            asst_dict = {"role": "assistant", "content": assistant_msg.content}
            if assistant_msg.tool_calls:
                asst_dict["tool_calls"] = [tc.to_dict() for tc in assistant_msg.tool_calls]

            messages.append(asst_dict)

            if not assistant_msg.tool_calls or choice.finish_reason == "stop":
                return response

            # Execute tools
            tool_messages = self.execute_tool_calls(assistant_msg.tool_calls)
            messages.extend(tool_messages)

        return response


# =============================================================================
# Conversation Manager
# =============================================================================

class Conversation:
    """Manages a multi-turn conversation with history."""

    def __init__(
        self,
        client=None,
        model: str = None,
        system: str = None,
        tools: List[dict] = None,
        tool_executor: ToolExecutor = None,
        max_history: int = None,
        **default_params,
    ):
        self.client = client
        self.model = model
        self.tools = tools
        self.tool_executor = tool_executor
        self.max_history = max_history
        self.default_params = default_params
        self.messages: List[dict] = []
        self._total_tokens = 0

        if system:
            self.messages.append({"role": "system", "content": system})

    def add_message(self, role: str, content: Union[str, list], **kwargs) -> "Conversation":
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self.messages.append(msg)
        self._trim_history()
        return self

    def add_user(self, content: Union[str, list], **kwargs) -> "Conversation":
        return self.add_message("user", content, **kwargs)

    def add_assistant(self, content: str, **kwargs) -> "Conversation":
        return self.add_message("assistant", content, **kwargs)

    def add_system(self, content: str) -> "Conversation":
        return self.add_message("system", content)

    def add_image(self, text: str, image_url: str, detail: str = "auto") -> "Conversation":
        msg = MessageBuilder.image_url(text, image_url, detail)
        self.messages.append(msg)
        self._trim_history()
        return self

    def _trim_history(self):
        if self.max_history is None:
            return
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        non_system = [m for m in self.messages if m.get("role") != "system"]
        if len(non_system) > self.max_history:
            non_system = non_system[-self.max_history:]
        self.messages = system_msgs + non_system

    def send(self, content: Union[str, list] = None, stream: bool = False, **kwargs) -> Union[ChatCompletion, Stream]:
        """Send a message and get a response."""
        if content is not None:
            self.add_user(content)

        params = {**self.default_params, **kwargs}
        if self.tools:
            params["tools"] = self.tools

        if self.tool_executor and self.tools:
            response = self.tool_executor.run_conversation(
                self.client, self.model, self.messages, **params
            )
            # Add the final response to our history (tool loop already added messages)
            # We need to sync messages
            return response

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=stream,
            **params,
        )

        if stream:
            return response

        choice = response.choices[0]
        asst_msg = {"role": "assistant", "content": choice.message.content}
        if choice.message.tool_calls:
            asst_msg["tool_calls"] = [tc.to_dict() for tc in choice.message.tool_calls]
        self.messages.append(asst_msg)

        if response.usage:
            self._total_tokens += response.usage.total_tokens

        return response

    def chat(self, content: str, **kwargs) -> str:
        """Simple chat that returns just the text content."""
        response = self.send(content, stream=False, **kwargs)
        return response.choices[0].message.content or ""

    @property
    def last_message(self) -> Optional[dict]:
        return self.messages[-1] if self.messages else None

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def clear(self, keep_system: bool = True):
        if keep_system:
            self.messages = [m for m in self.messages if m.get("role") == "system"]
        else:
            self.messages = []
        self._total_tokens = 0

    def fork(self) -> "Conversation":
        """Create a copy of this conversation."""
        new = Conversation(
            client=self.client,
            model=self.model,
            tools=self.tools,
            tool_executor=self.tool_executor,
            max_history=self.max_history,
            **self.default_params,
        )
        new.messages = copy.deepcopy(self.messages)
        new._total_tokens = self._total_tokens
        return new

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "messages": self.messages,
            "total_tokens": self._total_tokens,
        }

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str, client=None, **kwargs) -> "Conversation":
        with open(filepath) as f:
            data = json.load(f)
        conv = cls(client=client, model=data.get("model"), **kwargs)
        conv.messages = data.get("messages", [])
        conv._total_tokens = data.get("total_tokens", 0)
        return conv


# =============================================================================
# Response Parsers
# =============================================================================

def _parse_message(data: dict) -> ChatMessage:
    """Parse a message dict into a ChatMessage object."""
    tool_calls = None
    if data.get("tool_calls"):
        tool_calls = []
        for tc_data in data["tool_calls"]:
            func_data = tc_data.get("function", {})
            tc = ToolCall(
                id=tc_data.get("id"),
                type=tc_data.get("type", "function"),
                function=FunctionCall(
                    name=func_data.get("name"),
                    arguments=func_data.get("arguments"),
                ),
            )
            tool_calls.append(tc)

    function_call = None
    if data.get("function_call"):
        fc_data = data["function_call"]
        function_call = FunctionCall(
            name=fc_data.get("name"),
            arguments=fc_data.get("arguments"),
        )

    return ChatMessage(
        role=data.get("role"),
        content=data.get("content"),
        name=data.get("name"),
        tool_calls=tool_calls,
        function_call=function_call,
        tool_call_id=data.get("tool_call_id"),
        refusal=data.get("refusal"),
    )


def _parse_delta(data: dict) -> DeltaMessage:
    """Parse a delta dict into a DeltaMessage."""
    tool_calls = None
    if data.get("tool_calls"):
        tool_calls = []
        for tc_data in data["tool_calls"]:
            func_data = tc_data.get("function", {})
            tc = ToolCall(
                id=tc_data.get("id"),
                type=tc_data.get("type", "function"),
                function=FunctionCall(
                    name=func_data.get("name"),
                    arguments=func_data.get("arguments"),
                ),
            )
            if "index" in tc_data:
                tc.index = tc_data["index"]
            tool_calls.append(tc)

    function_call = None
    if data.get("function_call"):
        fc_data = data["function_call"]
        function_call = FunctionCall(
            name=fc_data.get("name"),
            arguments=fc_data.get("arguments"),
        )

    return DeltaMessage(
        role=data.get("role"),
        content=data.get("content"),
        tool_calls=tool_calls,
        function_call=function_call,
        refusal=data.get("refusal"),
    )


def _parse_logprobs(data: dict) -> Optional[ChoiceLogprobs]:
    if not data:
        return None
    content = data.get("content")
    if content is None:
        return None
    parsed_content = []
    for item in content:
        top_lps = None
        if item.get("top_logprobs"):
            top_lps = [TopLogprob(**tlp) for tlp in item["top_logprobs"]]
        parsed_content.append(TokenLogprob(
            token=item.get("token"),
            logprob=item.get("logprob"),
            bytes=item.get("bytes"),
            top_logprobs=top_lps,
        ))
    return ChoiceLogprobs(content=parsed_content)


def _parse_usage(data: dict) -> Optional[Usage]:
    if not data:
        return None
    return Usage(
        prompt_tokens=data.get("prompt_tokens", 0),
        completion_tokens=data.get("completion_tokens", 0),
        total_tokens=data.get("total_tokens", 0),
    )


def _parse_completion(data: dict) -> ChatCompletion:
    """Parse a completion response dict."""
    choices = []
    for c_data in data.get("choices", []):
        msg = _parse_message(c_data.get("message", {}))
        logprobs = _parse_logprobs(c_data.get("logprobs"))
        choices.append(Choice(
            index=c_data.get("index", 0),
            message=msg,
            finish_reason=c_data.get("finish_reason"),
            logprobs=logprobs,
        ))

    return ChatCompletion(
        id=data.get("id"),
        object=data.get("object", "chat.completion"),
        created=data.get("created"),
        model=data.get("model"),
        choices=choices,
        usage=_parse_usage(data.get("usage")),
        system_fingerprint=data.get("system_fingerprint"),
        service_tier=data.get("service_tier"),
    )


def _parse_chunk(data: dict) -> ChatCompletionChunk:
    """Parse a streaming chunk dict."""
    choices = []
    for c_data in data.get("choices", []):
        delta = _parse_delta(c_data.get("delta", {}))
        logprobs = _parse_logprobs(c_data.get("logprobs"))
        choices.append(StreamChoice(
            index=c_data.get("index", 0),
            delta=delta,
            finish_reason=c_data.get("finish_reason"),
            logprobs=logprobs,
        ))

    return ChatCompletionChunk(
        id=data.get("id"),
        object=data.get("object", "chat.completion.chunk"),
        created=data.get("created"),
        model=data.get("model"),
        choices=choices,
        usage=_parse_usage(data.get("usage")),
        system_fingerprint=data.get("system_fingerprint"),
        service_tier=data.get("service_tier"),
    )


# =============================================================================
# Resource Classes (Namespace Emulation)
# =============================================================================

class Completions:
    """chat.completions resource - the core of this library."""

    def __init__(self, transport: HTTPTransport, chat_path: str = "/chat/completions"):
        self._transport = transport
        self._chat_path = chat_path

    def create(
        self,
        *,
        model: str,
        messages: List[dict],
        frequency_penalty: float = None,
        function_call: Union[str, dict] = None,
        functions: List[dict] = None,
        logit_bias: Dict[str, float] = None,
        logprobs: bool = None,
        top_logprobs: int = None,
        max_tokens: int = None,
        max_completion_tokens: int = None,
        n: int = None,
        presence_penalty: float = None,
        response_format: dict = None,
        seed: int = None,
        stop: Union[str, List[str]] = None,
        stream: bool = False,
        stream_options: dict = None,
        temperature: float = None,
        top_p: float = None,
        tools: List[dict] = None,
        tool_choice: Union[str, dict] = None,
        parallel_tool_calls: bool = None,
        user: str = None,
        # Extra/custom parameters
        extra_body: dict = None,
        extra_headers: dict = None,
        timeout: float = None,
        **kwargs,
    ) -> Union[ChatCompletion, Stream]:
        """Create a chat completion."""
        body = {"model": model, "messages": messages}

        # Add optional parameters
        optional_params = {
            "frequency_penalty": frequency_penalty,
            "function_call": function_call,
            "functions": functions,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "n": n,
            "presence_penalty": presence_penalty,
            "response_format": response_format,
            "seed": seed,
            "stop": stop,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "user": user,
        }

        if stream and stream_options:
            body["stream_options"] = stream_options

        for key, value in optional_params.items():
            if value is not None:
                body[key] = value

        # Extra body params
        if extra_body:
            body.update(extra_body)

        # Extra kwargs
        body.update(kwargs)

        # Save/restore timeout
        original_timeout = self._transport.timeout
        if timeout is not None:
            self._transport.timeout = timeout

        try:
            if stream:
                raw_stream = self._transport.request(
                    "POST", self._chat_path, body=body, headers=extra_headers, stream=True
                )
                return Stream(self._parse_stream(raw_stream))
            else:
                response_data = self._transport.request(
                    "POST", self._chat_path, body=body, headers=extra_headers
                )
                return _parse_completion(response_data)
        finally:
            self._transport.timeout = original_timeout

    def _parse_stream(self, raw_stream: Iterator[str]) -> Iterator[ChatCompletionChunk]:
        """Parse SSE stream into ChatCompletionChunk objects."""
        for line in raw_stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    return
                try:
                    data = json.loads(data_str)
                    yield _parse_chunk(data)
                except json.JSONDecodeError:
                    continue
            elif line.startswith("{"):
                # Some servers send raw JSON without SSE prefix
                try:
                    data = json.loads(line)
                    yield _parse_chunk(data)
                except json.JSONDecodeError:
                    continue


class Chat:
    """chat resource namespace."""

    def __init__(self, transport: HTTPTransport, chat_path: str = "/chat/completions"):
        self.completions = Completions(transport, chat_path)


class Images:
    """Images resource - uses chat/completions under the hood for generation/editing/description."""

    def __init__(self, transport: HTTPTransport, chat_path: str = "/chat/completions", direct_path: str = "/images/generations"):
        self._transport = transport
        self._chat_path = chat_path
        self._direct_path = direct_path

    def generate(
        self,
        *,
        prompt: str,
        model: str = None,
        n: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = None,
        response_format: str = None,
        user: str = None,
        # If True, try the direct /images/generations endpoint first
        use_direct_endpoint: bool = True,
        **kwargs,
    ) -> ImageResponse:
        """Generate images. Tries direct endpoint first, falls back to chat completions."""
        if use_direct_endpoint:
            try:
                body = {"prompt": prompt, "n": n, "size": size}
                if model:
                    body["model"] = model
                if quality:
                    body["quality"] = quality
                if style:
                    body["style"] = style
                if response_format:
                    body["response_format"] = response_format
                if user:
                    body["user"] = user
                body.update(kwargs)

                response_data = self._transport.request("POST", self._direct_path, body=body)
                return self._parse_image_response(response_data)
            except (NotFoundError, APIError):
                pass

        # Fallback: use chat completions to generate image description/URL
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an image generation assistant. Generate {n} image(s) based on the user's prompt. "
                    f"Size: {size}, Quality: {quality}. "
                    "Respond with a JSON object containing a 'data' array of objects with 'url' or 'b64_json' and 'revised_prompt' keys."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        body = {
            "model": model or "default",
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        body.update(kwargs)

        response_data = self._transport.request("POST", self._chat_path, body=body)
        completion = _parse_completion(response_data)

        # Try to parse the response as image data
        content = completion.choices[0].message.content if completion.choices else ""
        try:
            img_data = json.loads(content)
            if "data" in img_data:
                return self._parse_image_response(img_data)
        except (json.JSONDecodeError, KeyError):
            pass

        # Return raw content as revised_prompt
        return ImageResponse(data=[ImageData(revised_prompt=content)])

    def edit(
        self,
        *,
        image: Union[str, bytes],
        prompt: str,
        model: str = None,
        mask: Union[str, bytes] = None,
        n: int = 1,
        size: str = "1024x1024",
        **kwargs,
    ) -> ImageResponse:
        """Edit an image using chat completions with vision."""
        if isinstance(image, str) and os.path.exists(image):
            with open(image, "rb") as f:
                image = f.read()

        if isinstance(image, bytes):
            b64_image = base64.b64encode(image).decode("utf-8")
            image_url = f"data:image/png;base64,{b64_image}"
        else:
            image_url = image

        messages = [
            {
                "role": "system",
                "content": f"You are an image editing assistant. Edit the provided image according to the user's instructions. Respond with the edited image data.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

        body = {"model": model or "default", "messages": messages}
        body.update(kwargs)

        response_data = self._transport.request("POST", self._chat_path, body=body)
        completion = _parse_completion(response_data)
        content = completion.choices[0].message.content if completion.choices else ""

        return ImageResponse(data=[ImageData(revised_prompt=content)])

    def describe(
        self,
        *,
        image: Union[str, bytes],
        model: str = None,
        prompt: str = "Describe this image in detail.",
        detail: str = "high",
        **kwargs,
    ) -> str:
        """Describe an image using vision capabilities."""
        if isinstance(image, str) and os.path.exists(image):
            with open(image, "rb") as f:
                image = f.read()

        if isinstance(image, bytes):
            b64_image = base64.b64encode(image).decode("utf-8")
            image_url = f"data:image/png;base64,{b64_image}"
        else:
            image_url = image

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": detail}},
                ],
            },
        ]

        body = {"model": model or "default", "messages": messages}
        body.update(kwargs)

        response_data = self._transport.request("POST", self._chat_path, body=body)
        completion = _parse_completion(response_data)
        return completion.choices[0].message.content if completion.choices else ""

    def _parse_image_response(self, data: dict) -> ImageResponse:
        images = []
        for item in data.get("data", []):
            images.append(ImageData(
                url=item.get("url"),
                b64_json=item.get("b64_json"),
                revised_prompt=item.get("revised_prompt"),
            ))
        return ImageResponse(
            created=data.get("created", int(time.time())),
            data=images,
        )


class Embeddings:
    """Embeddings resource - uses chat/completions to simulate embeddings if direct endpoint unavailable."""

    def __init__(self, transport: HTTPTransport, chat_path: str = "/chat/completions", direct_path: str = "/embeddings"):
        self._transport = transport
        self._chat_path = chat_path
        self._direct_path = direct_path

    def create(
        self,
        *,
        input: Union[str, List[str]],
        model: str,
        encoding_format: str = "float",
        dimensions: int = None,
        user: str = None,
        use_direct_endpoint: bool = True,
        **kwargs,
    ) -> EmbeddingResponse:
        """Create embeddings. Tries direct endpoint first, falls back to chat-based simulation."""
        if isinstance(input, str):
            input = [input]

        if use_direct_endpoint:
            try:
                body = {"input": input, "model": model}
                if encoding_format:
                    body["encoding_format"] = encoding_format
                if dimensions:
                    body["dimensions"] = dimensions
                if user:
                    body["user"] = user
                body.update(kwargs)

                response_data = self._transport.request("POST", self._direct_path, body=body)
                return self._parse_embedding_response(response_data)
            except (NotFoundError, APIError):
                pass

        # Fallback: Use chat completions to generate pseudo-embeddings
        dim = dimensions or 1536
        results = []
        for i, text in enumerate(input):
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"Generate a numerical embedding vector of dimension {dim} for the following text. "
                        f"Respond with only a JSON array of {dim} floating point numbers between -1 and 1."
                    ),
                },
                {"role": "user", "content": text},
            ]
            body = {
                "model": model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0,
            }

            response_data = self._transport.request("POST", self._chat_path, body=body)
            completion = _parse_completion(response_data)
            content = completion.choices[0].message.content if completion.choices else "[]"

            try:
                embedding = json.loads(content)
                if isinstance(embedding, dict):
                    embedding = list(embedding.values())[0] if embedding else []
                if not isinstance(embedding, list):
                    embedding = self._text_to_hash_embedding(text, dim)
            except json.JSONDecodeError:
                embedding = self._text_to_hash_embedding(text, dim)

            results.append(EmbeddingData(
                embedding=embedding[:dim],
                index=i,
            ))

        return EmbeddingResponse(
            data=results,
            model=model,
            usage=Usage(
                prompt_tokens=sum(len(t.split()) for t in input),
                total_tokens=sum(len(t.split()) for t in input),
            ),
        )

    def _text_to_hash_embedding(self, text: str, dim: int) -> List[float]:
        """Generate a deterministic pseudo-embedding from text using hashing."""
        result = []
        for i in range(dim):
            h = hashlib.sha256(f"{text}:{i}".encode()).digest()
            val = struct.unpack('f', h[:4])[0]
            # Normalize to [-1, 1]
            val = (val % 2.0) - 1.0
            result.append(round(val, 6))
        return result

    def _parse_embedding_response(self, data: dict) -> EmbeddingResponse:
        results = []
        for item in data.get("data", []):
            results.append(EmbeddingData(
                embedding=item.get("embedding", []),
                index=item.get("index", 0),
            ))
        return EmbeddingResponse(
            data=results,
            model=data.get("model"),
            usage=_parse_usage(data.get("usage")),
        )


class Audio:
    """Audio resource namespace."""

    def __init__(self, transport: HTTPTransport, chat_path: str = "/chat/completions"):
        self._transport = transport
        self._chat_path = chat_path
        self.transcriptions = AudioTranscriptions(transport, chat_path)
        self.translations = AudioTranslations(transport, chat_path)
        self.speech = AudioSpeech(transport, chat_path)


class AudioTranscriptions:
    """Audio transcription resource."""

    def __init__(self, transport: HTTPTransport, chat_path: str = "/chat/completions"):
        self._transport = transport
        self._chat_path = chat_path

    def create(
        self,
        *,
        file: Union[str, bytes, Any],
        model: str,
        language: str = None,
        prompt: str = None,
        response_format: str = "json",
        temperature: float = None,
        timestamp_granularities: List[str] = None,
        use_direct_endpoint: bool = True,
        **kwargs,
    ) -> AudioTranscription:
        """Transcribe audio. Tries direct endpoint, falls back to chat completions with audio."""
        if use_direct_endpoint:
            try:
                fields = {"model": model}
                if language:
                    fields["language"] = language
                if prompt:
                    fields["prompt"] = prompt
                if response_format:
                    fields["response_format"] = response_format
                if temperature is not None:
                    fields["temperature"] = str(temperature)

                files_dict = {"file": file}
                response_data = self._transport.multipart_request(
                    "POST", "/audio/transcriptions", fields=fields, files=files_dict
                )

                if isinstance(response_data, dict):
                    return AudioTranscription(
                        text=response_data.get("text", response_data.get("raw", "")),
                        language=response_data.get("language"),
                        duration=response_data.get("duration"),
                        segments=response_data.get("segments"),
                    )
            except (NotFoundError, APIError):
                pass

        # Fallback: Use chat completions
        if isinstance(file, str) and os.path.exists(file):
            with open(file, "rb") as f:
                audio_data = f.read()
        elif isinstance(file, bytes):
            audio_data = file
        else:
            audio_data = file.read()

        b64_audio = base64.b64encode(audio_data).decode("utf-8")

        messages = [
            {
                "role": "system",
                "content": "You are an audio transcription assistant. Transcribe the provided audio accurately.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Please transcribe this audio."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64_audio, "format": "wav"},
                    },
                ],
            },
        ]

        body = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature

        response_data = self._transport.request("POST", self._chat_path, body=body)
        completion = _parse_completion(response_data)
        text = completion.choices[0].message.content if completion.choices else ""

        return AudioTranscription(text=text)


class AudioTranslations:
    """Audio translation resource."""

    def __init__(self, transport: HTTPTransport, chat_path: str = "/chat/completions"):
        self._transport = transport
        self._chat_path = chat_path

    def create(
        self,
        *,
        file: Union[str, bytes, Any],
        model: str,
        prompt: str = None,
        response_format: str = "json",
        temperature: float = None,
        use_direct_endpoint: bool = True,
        **kwargs,
    ) -> AudioTranslation:
        """Translate audio to English."""
        if use_direct_endpoint:
            try:
                fields = {"model": model}
                if prompt:
                    fields["prompt"] = prompt
                if response_format:
                    fields["response_format"] = response_format
                if temperature is not None:
                    fields["temperature"] = str(temperature)

                files_dict = {"file": file}
                response_data = self._transport.multipart_request(
                    "POST", "/audio/translations", fields=fields, files=files_dict
                )

                if isinstance(response_data, dict):
                    return AudioTranslation(
                        text=response_data.get("text", response_data.get("raw", ""))
                    )
            except (NotFoundError, APIError):
                pass

        # Fallback: Use chat completions
        if isinstance(file, str) and os.path.exists(file):
            with open(file, "rb") as f:
                audio_data = f.read()
        elif isinstance(file, bytes):
            audio_data = file
        else:
            audio_data = file.read()

        b64_audio = base64.b64encode(audio_data).decode("utf-8")

        messages = [
            {
                "role": "system",
                "content": "You are a translation assistant. Translate the provided audio content to English.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Translate this audio to English."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64_audio, "format": "wav"},
                    },
                ],
            },
        ]

        body = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature

        response_data = self._transport.request("POST", self._chat_path, body=body)
        completion = _parse_completion(response_data)
        text = completion.choices[0].message.content if completion.choices else ""

        return AudioTranslation(text=text)


class AudioSpeech:
    """Text-to-speech resource."""

    def __init__(self, transport: HTTPTransport, chat_path: str = "/chat/completions"):
        self._transport = transport
        self._chat_path = chat_path

    def create(
        self,
        *,
        model: str,
        input: str,
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
        use_direct_endpoint: bool = True,
        **kwargs,
    ) -> AudioSpeechResponse:
        """Generate speech from text."""
        if use_direct_endpoint:
            try:
                body = {
                    "model": model,
                    "input": input,
                    "voice": voice,
                    "response_format": response_format,
                    "speed": speed,
                }
                body.update(kwargs)
                raw_bytes = self._transport.request_raw("POST", "/audio/speech", body=body)
                return AudioSpeechResponse(content=raw_bytes)
            except (NotFoundError, APIError):
                pass

        # Fallback: Use chat to describe what would be spoken
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a text-to-speech assistant. Voice: {voice}, Speed: {speed}x, Format: {response_format}. "
                    "Since direct TTS is not available, provide an SSML or phonetic representation of how the following text should be spoken."
                ),
            },
            {"role": "user", "content": input},
        ]

        body_chat = {"model": model, "messages": messages}
        response_data = self._transport.request("POST", self._chat_path, body=body_chat)
        completion = _parse_completion(response_data)
        text = completion.choices[0].message.content if completion.choices else input

        return AudioSpeechResponse(content=text.encode("utf-8"))


class Moderations:
    """Content moderation resource."""

    def __init__(self, transport: HTTPTransport, chat_path: str = "/chat/completions"):
        self._transport = transport
        self._chat_path = chat_path

    def create(
        self,
        *,
        input: Union[str, List[str]],
        model: str = "text-moderation-latest",
        use_direct_endpoint: bool = True,
        **kwargs,
    ) -> ModerationResponse:
        """Moderate content."""
        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = input

        if use_direct_endpoint:
            try:
                body = {"input": inputs, "model": model}
                body.update(kwargs)
                response_data = self._transport.request("POST", "/moderations", body=body)
                return self._parse_moderation_response(response_data)
            except (NotFoundError, APIError):
                pass

        # Fallback: Use chat completions for moderation
        categories = [
            "hate", "hate/threatening", "harassment", "harassment/threatening",
            "self-harm", "self-harm/intent", "self-harm/instructions",
            "sexual", "sexual/minors", "violence", "violence/graphic",
        ]

        results = []
        for text in inputs:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a content moderation system. Analyze the following text for policy violations. "
                        f"Respond with a JSON object containing: 'flagged' (boolean), "
                        f"'categories' (object with keys: {', '.join(categories)} each boolean), "
                        f"'category_scores' (object with same keys, each float 0-1)."
                    ),
                },
                {"role": "user", "content": text},
            ]

            body = {
                "model": model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0,
            }

            try:
                response_data = self._transport.request("POST", self._chat_path, body=body)
                completion = _parse_completion(response_data)
                content = completion.choices[0].message.content if completion.choices else "{}"
                mod_data = json.loads(content)

                results.append(ModerationResult(
                    flagged=mod_data.get("flagged", False),
                    categories=mod_data.get("categories", {}),
                    category_scores=mod_data.get("category_scores", {}),
                ))
            except (json.JSONDecodeError, APIError):
                results.append(ModerationResult(flagged=False))

        return ModerationResponse(
            model=model,
            results=results,
        )

    def _parse_moderation_response(self, data: dict) -> ModerationResponse:
        results = []
        for r in data.get("results", []):
            results.append(ModerationResult(
                flagged=r.get("flagged", False),
                categories=r.get("categories", {}),
                category_scores=r.get("category_scores", {}),
            ))
        return ModerationResponse(
            id=data.get("id"),
            model=data.get("model"),
            results=results,
        )


class Models:
    """Models resource."""

    def __init__(self, transport: HTTPTransport):
        self._transport = transport

    def list(self, **kwargs) -> ModelList:
        """List available models."""
        try:
            response_data = self._transport.request("GET", "/models")
            models = []
            for m in response_data.get("data", []):
                models.append(ModelInfo(
                    id=m.get("id"),
                    object=m.get("object", "model"),
                    created=m.get("created"),
                    owned_by=m.get("owned_by"),
                ))
            return ModelList(data=models)
        except APIError:
            return ModelList(data=[])

    def retrieve(self, model: str, **kwargs) -> ModelInfo:
        """Retrieve a specific model."""
        try:
            response_data = self._transport.request("GET", f"/models/{model}")
            return ModelInfo(
                id=response_data.get("id"),
                object=response_data.get("object", "model"),
                created=response_data.get("created"),
                owned_by=response_data.get("owned_by"),
            )
        except APIError as e:
            raise e

    def delete(self, model: str, **kwargs) -> dict:
        """Delete a fine-tuned model."""
        try:
            return self._transport.request("DELETE", f"/models/{model}")
        except APIError as e:
            raise e


class Completions_Legacy:
    """Legacy /completions endpoint support through chat completions."""

    def __init__(self, transport: HTTPTransport, chat_path: str = "/chat/completions"):
        self._transport = transport
        self._chat_path = chat_path

    def create(
        self,
        *,
        model: str,
        prompt: Union[str, List[str]] = "<|endoftext|>",
        suffix: str = None,
        max_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        logprobs: int = None,
        echo: bool = False,
        stop: Union[str, List[str]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        best_of: int = None,
        logit_bias: dict = None,
        user: str = None,
        **kwargs,
    ) -> ChatCompletion:
        """Create a legacy completion using chat completions endpoint."""
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)

        messages = [{"role": "user", "content": prompt}]

        if suffix:
            messages[0]["content"] += f"\n[Continue with suffix: {suffix}]"

        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }

        if logit_bias:
            body["logit_bias"] = logit_bias
        if user:
            body["user"] = user

        body.update(kwargs)

        # Remove None values
        body = {k: v for k, v in body.items() if v is not None}

        if stream:
            raw_stream = self._transport.request("POST", self._chat_path, body=body, stream=True)
            completions_obj = Completions(self._transport, self._chat_path)
            return Stream(completions_obj._parse_stream(raw_stream))

        response_data = self._transport.request("POST", self._chat_path, body=body)
        return _parse_completion(response_data)


# =============================================================================
# Batch Processing
# =============================================================================

class BatchProcessor:
    """Process multiple requests in batch."""

    def __init__(self, client, max_concurrent: int = 5, rate_limit_delay: float = 0.1):
        self.client = client
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay

    def process(
        self,
        requests: List[dict],
        model: str = None,
        **default_params,
    ) -> List[Union[ChatCompletion, APIError]]:
        """Process a batch of chat completion requests."""
        results = [None] * len(requests)
        errors = [None] * len(requests)

        semaphore = threading.Semaphore(self.max_concurrent)
        threads = []

        def make_request(index: int, request_params: dict):
            try:
                semaphore.acquire()
                time.sleep(self.rate_limit_delay)

                params = {**default_params, **request_params}
                if model and "model" not in params:
                    params["model"] = model

                result = self.client.chat.completions.create(**params)
                results[index] = result
            except APIError as e:
                errors[index] = e
                results[index] = e
            finally:
                semaphore.release()

        for i, req in enumerate(requests):
            t = threading.Thread(target=make_request, args=(i, req))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results

    def map(
        self,
        prompts: List[str],
        model: str,
        system: str = None,
        **params,
    ) -> List[str]:
        """Simple batch: map a list of prompts to responses."""
        requests = []
        for prompt in prompts:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            requests.append({"messages": messages, "model": model, **params})

        results = self.process(requests)
        outputs = []
        for r in results:
            if isinstance(r, APIError):
                outputs.append(f"[ERROR: {r.message}]")
            elif isinstance(r, ChatCompletion):
                outputs.append(r.choices[0].message.content if r.choices else "")
            else:
                outputs.append("")
        return outputs


# =============================================================================
# Retry Decorator
# =============================================================================

def with_retry(max_retries: int = 3, backoff_factor: float = 1.0, retry_on: tuple = (RateLimitError, InternalServerError, APIConnectionError)):
    """Decorator to add retry logic to any function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_error = e
                    if attempt < max_retries:
                        wait = backoff_factor * (2 ** attempt)
                        time.sleep(wait)
            raise last_error
        return wrapper
    return decorator


# =============================================================================
# Main Client
# =============================================================================

class Client:
    """
    Main API client - mirrors OpenAI Python SDK interface.
    
    Usage:
        client = Client(api_key="sk-...", base_url="https://api.openai.com/v1")
        
        # Chat
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        
        # Streaming
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True
        )
        for chunk in stream:
            print(chunk.choices[0].delta.content, end="")
        
        # Images
        response = client.images.generate(prompt="A cat")
        
        # Embeddings
        response = client.embeddings.create(input="Hello", model="text-embedding-ada-002")
        
        # Audio
        transcript = client.audio.transcriptions.create(file="audio.wav", model="whisper-1")
    """

    def __init__(
        self,
        api_key: str = None,
        organization: str = None,
        base_url: str = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        default_headers: dict = None,
        default_query: dict = None,
        verify_ssl: bool = True,
        proxy: str = None,
        # Aliases
        key: str = None,
        url: str = None,
        org: str = None,
    ):
        # Resolve parameters with aliases and env vars
        self.api_key = api_key or key or os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", ""))
        self.organization = organization or org or os.environ.get("OPENAI_ORG_ID", os.environ.get("OPENAI_ORGANIZATION"))
        self.base_url = (base_url or url or os.environ.get("OPENAI_BASE_URL", os.environ.get("API_BASE_URL", "https://api.openai.com/v1"))).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_headers = default_headers or {}
        self.verify_ssl = verify_ssl
        self.proxy = proxy

        # Create transport
        self._transport = HTTPTransport(
            base_url=self.base_url,
            api_key=self.api_key,
            organization=self.organization,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            verify_ssl=self.verify_ssl,
            proxy=self.proxy,
        )

        # Initialize resource namespaces
        self.chat = Chat(self._transport)
        self.images = Images(self._transport)
        self.embeddings = Embeddings(self._transport)
        self.audio = Audio(self._transport)
        self.moderations = Moderations(self._transport)
        self.models = Models(self._transport)
        self.completions = Completions_Legacy(self._transport)

        # Helpers
        self.message = MessageBuilder
        self.tools = ToolDefinition
        self.batch = BatchProcessor(self)

    def conversation(
        self,
        model: str = None,
        system: str = None,
        tools: List[dict] = None,
        tool_executor: ToolExecutor = None,
        max_history: int = None,
        **kwargs,
    ) -> Conversation:
        """Create a new conversation manager."""
        return Conversation(
            client=self,
            model=model,
            system=system,
            tools=tools,
            tool_executor=tool_executor,
            max_history=max_history,
            **kwargs,
        )

    def quick(
        self,
        prompt: str,
        model: str = None,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs,
    ) -> str:
        """Quick one-shot completion - returns just the text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params = {"model": model or "default", "messages": messages}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.update(kwargs)

        response = self.chat.completions.create(**params)
        return response.choices[0].message.content if response.choices else ""

    def quick_json(
        self,
        prompt: str,
        model: str = None,
        system: str = None,
        schema: dict = None,
        **kwargs,
    ) -> dict:
        """Quick one-shot completion that returns parsed JSON."""
        sys_msg = system or "Respond with valid JSON only."
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]

        params = {
            "model": model or "default",
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if schema:
            params["response_format"] = ToolDefinition.response_format_json("response", schema)
        params.update(kwargs)

        response = self.chat.completions.create(**params)
        content = response.choices[0].message.content if response.choices else "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw": content}

    def quick_stream(
        self,
        prompt: str,
        model: str = None,
        system: str = None,
        callback: Callable[[str], None] = None,
        **kwargs,
    ) -> str:
        """Quick streaming completion. Optionally calls callback with each token."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = self.chat.completions.create(
            model=model or "default",
            messages=messages,
            stream=True,
            **kwargs,
        )

        full_content = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_content.append(token)
                if callback:
                    callback(token)

        return "".join(full_content)

    def vision(
        self,
        prompt: str,
        image: Union[str, bytes, List[str]],
        model: str = None,
        detail: str = "auto",
        **kwargs,
    ) -> str:
        """Quick vision query with image(s)."""
        if isinstance(image, list):
            msg = MessageBuilder.multi_image(prompt, image, detail)
        elif isinstance(image, bytes):
            b64 = base64.b64encode(image).decode("utf-8")
            msg = MessageBuilder.image_base64(prompt, b64, detail=detail)
        elif isinstance(image, str) and os.path.exists(image):
            msg = MessageBuilder.image_file(prompt, image, detail)
        else:
            msg = MessageBuilder.image_url(prompt, image, detail)

        response = self.chat.completions.create(
            model=model or "default",
            messages=[msg],
            **kwargs,
        )
        return response.choices[0].message.content if response.choices else ""

    def tool_executor(self) -> ToolExecutor:
        """Create a new ToolExecutor instance."""
        return ToolExecutor()

    def __repr__(self):
        return f"Client(base_url={self.base_url!r}, api_key={'***' if self.api_key else None!r})"

    def close(self):
        """Close the client (no-op for urllib, but kept for compatibility)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# Convenience Aliases
# =============================================================================

# OpenAI-compatible alias
OpenAI = Client

# Quick function aliases
def client(api_key: str = None, base_url: str = None, **kwargs) -> Client:
    """Create a new client."""
    return Client(api_key=api_key, base_url=base_url, **kwargs)


def chat(
    prompt: str,
    model: str = "default",
    api_key: str = None,
    base_url: str = None,
    system: str = None,
    **kwargs,
) -> str:
    """Quick standalone chat function."""
    c = Client(api_key=api_key, base_url=base_url)
    return c.quick(prompt, model=model, system=system, **kwargs)


# =============================================================================
# Async Support (using threading)
# =============================================================================

class AsyncCompletions:
    """Async wrapper using threading for chat completions."""

    def __init__(self, completions: Completions):
        self._completions = completions

    def create(self, **kwargs) -> "AsyncResult":
        result = AsyncResult()

        def run():
            try:
                response = self._completions.create(**kwargs)
                result._set_result(response)
            except Exception as e:
                result._set_error(e)

        thread = threading.Thread(target=run)
        thread.start()
        return result


class AsyncResult:
    """Async result that can be awaited (via .result()) or used with callbacks."""

    def __init__(self):
        self._event = threading.Event()
        self._result = None
        self._error = None
        self._callbacks = []

    def _set_result(self, result):
        self._result = result
        self._event.set()
        for cb in self._callbacks:
            cb(result, None)

    def _set_error(self, error):
        self._error = error
        self._event.set()
        for cb in self._callbacks:
            cb(None, error)

    def result(self, timeout: float = None) -> Any:
        """Block until result is available."""
        self._event.wait(timeout=timeout)
        if self._error:
            raise self._error
        return self._result

    def then(self, callback: Callable) -> "AsyncResult":
        """Add a callback: callback(result, error)."""
        if self._event.is_set():
            callback(self._result, self._error)
        else:
            self._callbacks.append(callback)
        return self

    @property
    def done(self) -> bool:
        return self._event.is_set()


class AsyncChat:
    """Async chat namespace."""

    def __init__(self, chat: Chat):
        self.completions = AsyncCompletions(chat.completions)


class AsyncClient:
    """Async client wrapper using threading."""

    def __init__(self, *args, **kwargs):
        self._sync_client = Client(*args, **kwargs)
        self.chat = AsyncChat(self._sync_client.chat)
        self.models = self._sync_client.models
        self.images = self._sync_client.images
        self.embeddings = self._sync_client.embeddings
        self.audio = self._sync_client.audio
        self.moderations = self._sync_client.moderations
        self.completions = self._sync_client.completions
        self.message = MessageBuilder
        self.tools = ToolDefinition
        self.batch = self._sync_client.batch

    def quick(self, prompt: str, **kwargs) -> AsyncResult:
        result = AsyncResult()

        def run():
            try:
                text = self._sync_client.quick(prompt, **kwargs)
                result._set_result(text)
            except Exception as e:
                result._set_error(e)

        threading.Thread(target=run).start()
        return result

    def close(self):
        self._sync_client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# Utility Functions
# =============================================================================

def count_tokens_approx(text: str) -> int:
    """Approximate token count (roughly 4 chars per token for English)."""
    return max(1, len(text) // 4)


def count_messages_tokens_approx(messages: List[dict]) -> int:
    """Approximate token count for a list of messages."""
    total = 0
    for msg in messages:
        total += 4  # message overhead
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens_approx(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += count_tokens_approx(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    total += 85  # approximate tokens for image
        if msg.get("name"):
            total += count_tokens_approx(msg["name"])
    total += 2  # reply priming
    return total


def encode_image(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_to_data_url(image_path: str) -> str:
    """Convert an image file to a data URL."""
    media_type = mimetypes.guess_type(image_path)[0] or "image/png"
    b64 = encode_image(image_path)
    return f"data:{media_type};base64,{b64}"


def format_tool_calls_for_display(tool_calls: List[ToolCall]) -> str:
    """Format tool calls for human-readable display."""
    lines = []
    for tc in tool_calls:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            args_str = json.dumps(args, indent=2)
        except json.JSONDecodeError:
            args_str = tc.function.arguments
        lines.append(f"📞 {tc.function.name}({args_str})")
    return "\n".join(lines)


def merge_chunks(chunks: List[ChatCompletionChunk]) -> ChatCompletion:
    """Merge a list of streaming chunks into a single ChatCompletion."""
    stream = Stream(iter(chunks))
    # Force consume by iterating
    for _ in stream:
        pass
    return stream.collect()


# =============================================================================
# Export All
# =============================================================================

__all__ = [
    # Client
    "Client",
    "OpenAI",
    "AsyncClient",
    "client",
    "chat",
    # Models
    "BaseModel",
    "ChatMessage",
    "DeltaMessage",
    "Choice",
    "StreamChoice",
    "Usage",
    "ChatCompletion",
    "ChatCompletionChunk",
    "FunctionCall",
    "ToolCall",
    "TokenLogprob",
    "TopLogprob",
    "ChoiceLogprobs",
    "EmbeddingData",
    "EmbeddingResponse",
    "ImageData",
    "ImageResponse",
    "AudioTranscription",
    "AudioTranslation",
    "AudioSpeechResponse",
    "ModerationResult",
    "ModerationResponse",
    "ModelInfo",
    "ModelList",
    # Stream
    "Stream",
    # Helpers
    "MessageBuilder",
    "ToolDefinition",
    "ToolExecutor",
    "Conversation",
    "BatchProcessor",
    # Exceptions
    "APIError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "RateLimitError",
    "BadRequestError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    "ContentFilterError",
    # Utilities
    "count_tokens_approx",
    "count_messages_tokens_approx",
    "encode_image",
    "image_to_data_url",
    "format_tool_calls_for_display",
    "merge_chunks",
    "with_retry",
]


# =============================================================================
# Self-Test / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("API.py - Complete Single-File OpenAI-Compatible Client Library")
    print("=" * 70)
    print()
    print("Features:")
    print("  ✅ Chat Completions (streaming & non-streaming)")
    print("  ✅ Tool/Function Calling with auto-execution loop")
    print("  ✅ Vision/Image Understanding (multi-image support)")
    print("  ✅ Image Generation (direct + chat fallback)")
    print("  ✅ Embeddings (direct + chat fallback)")
    print("  ✅ Audio Transcription & Translation (direct + chat fallback)")
    print("  ✅ Text-to-Speech (direct + chat fallback)")
    print("  ✅ Content Moderation (direct + chat fallback)")
    print("  ✅ Model Listing & Retrieval")
    print("  ✅ Conversation Manager with history")
    print("  ✅ Batch Processing with concurrency control")
    print("  ✅ Message Builder helpers (text, image, audio)")
    print("  ✅ Tool Definition helpers (from dict or callable)")
    print("  ✅ JSON Mode / Structured Output")
    print("  ✅ Retry logic with exponential backoff")
    print("  ✅ SSE Streaming with Stream wrapper")
    print("  ✅ Async support (thread-based)")
    print("  ✅ Token counting utilities")
    print("  ✅ Comprehensive error hierarchy")
    print("  ✅ Zero dependencies (stdlib only)")
    print()
    print("Quick Start:")
    print("-" * 40)
    print("""
from API import Client

# Initialize
client = Client(
    api_key="your-key",
    base_url="http://localhost:8000/v1"  # Any OpenAI-compatible endpoint
)

# Simple chat
response = client.chat.completions.create(
    model="my-model",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)

# Quick one-liner
print(client.quick("What is 2+2?", model="my-model"))

# Streaming
for chunk in client.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# Tool calling
executor = client.tool_executor()

@executor.register
def get_weather(city: str, unit: str = "celsius"):
    \"\"\"Get the current weather for a city.\"\"\"
    return {"temp": 22, "unit": unit, "city": city}

response = executor.run_conversation(
    client, "my-model",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}]
)

# Vision
description = client.vision(
    "What's in this image?",
    image="https://example.com/photo.jpg",
    model="my-vision-model"
)

# Conversation with history
conv = client.conversation(model="my-model", system="You are a pirate.")
print(conv.chat("Hello!"))
print(conv.chat("What's your name?"))

# JSON mode
data = client.quick_json("List 3 colors", model="my-model")

# Batch processing
results = client.batch.map(
    prompts=["Hello", "Hi", "Hey"],
    model="my-model"
)
""")
    print()
    print("Environment Variables:")
    print("  OPENAI_API_KEY or API_KEY      - API key")
    print("  OPENAI_BASE_URL or API_BASE_URL - Base URL")
    print("  OPENAI_ORG_ID                   - Organization ID")
    print()

    # Quick functional test with a mock
    print("Running internal model tests...")

    # Test BaseModel
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.to_dict() == {"role": "user", "content": "Hello"}
    print("  ✅ ChatMessage")

    # Test ToolCall
    tc = ToolCall(
        id="call_123",
        function=FunctionCall(name="test", arguments='{"a": 1}')
    )
    assert tc.function.name == "test"
    d = tc.to_dict()
    assert d["function"]["name"] == "test"
    print("  ✅ ToolCall")

    # Test ToolDefinition
    td = ToolDefinition.function("get_weather", "Get weather", {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    })
    assert td["function"]["name"] == "get_weather"
    print("  ✅ ToolDefinition")

    # Test ToolExecutor
    executor = ToolExecutor()

    @executor.register
    def add(a: int, b: int):
        """Add two numbers."""
        return {"result": a + b}

    result = executor.execute("add", '{"a": 2, "b": 3}')
    assert json.loads(result)["result"] == 5
    assert len(executor.definitions) == 1
    print("  ✅ ToolExecutor")

    # Test function_from_callable
    def sample_func(name: str, age: int, active: bool = True):
        """A sample function."""
        pass

    td2 = ToolDefinition.function_from_callable(sample_func)
    assert td2["function"]["name"] == "sample_func"
    assert "name" in td2["function"]["parameters"]["properties"]
    assert "age" in td2["function"]["parameters"]["properties"]
    print("  ✅ ToolDefinition.function_from_callable")

    # Test MessageBuilder
    img_msg = MessageBuilder.image_url("Describe", "http://example.com/img.png")
    assert img_msg["role"] == "user"
    assert len(img_msg["content"]) == 2
    assert img_msg["content"][1]["type"] == "image_url"
    print("  ✅ MessageBuilder")

    # Test response parsing
    raw = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello there!",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "test_func",
                        "arguments": '{"x": 1}'
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
    completion = _parse_completion(raw)
    assert completion.id == "chatcmpl-test"
    assert completion.choices[0].message.content == "Hello there!"
    assert completion.choices[0].message.tool_calls[0].function.name == "test_func"
    assert completion.usage.total_tokens == 15
    print("  ✅ Response Parsing")

    # Test chunk parsing
    chunk_raw = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": "Hi"},
            "finish_reason": None
        }]
    }
    chunk = _parse_chunk(chunk_raw)
    assert chunk.choices[0].delta.content == "Hi"
    assert chunk.choices[0].delta.role == "assistant"
    print("  ✅ Chunk Parsing")

    # Test Stream.collect()
    chunks = [
        _parse_chunk({
            "id": "chatcmpl-1", "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        }),
        _parse_chunk({
            "id": "chatcmpl-1", "model": "m",
            "choices": [{"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}]
        }),
        _parse_chunk({
            "id": "chatcmpl-1", "model": "m",
            "choices": [{"index": 0, "delta": {"content": "world!"}, "finish_reason": "stop"}]
        }),
    ]
    stream = Stream(iter(chunks))
    collected = stream.collect()
    assert collected.choices[0].message.content == "Hello world!"
    assert collected.choices[0].finish_reason == "stop"
    print("  ✅ Stream.collect()")

    # Test token counting
    assert count_tokens_approx("Hello world") > 0
    assert count_messages_tokens_approx([{"role": "user", "content": "Hi"}]) > 0
    print("  ✅ Token Counting")

    # Test Conversation
    conv = Conversation(model="test")
    conv.add_system("You are helpful")
    conv.add_user("Hello")
    assert len(conv.messages) == 2
    assert conv.messages[0]["role"] == "system"
    forked = conv.fork()
    assert len(forked.messages) == 2
    forked.add_user("Another")
    assert len(conv.messages) == 2  # Original unchanged
    assert len(forked.messages) == 3
    print("  ✅ Conversation")

    # Test error hierarchy
    assert issubclass(AuthenticationError, APIError)
    assert issubclass(RateLimitError, APIError)
    assert issubclass(APITimeoutError, APIConnectionError)
    print("  ✅ Error Hierarchy")

    # Test Client init
    c = Client(api_key="test-key", base_url="http://localhost:8000/v1")
    assert c.api_key == "test-key"
    assert c.base_url == "http://localhost:8000/v1"
    assert hasattr(c, "chat")
    assert hasattr(c.chat, "completions")
    assert hasattr(c, "images")
    assert hasattr(c, "embeddings")
    assert hasattr(c, "audio")
    assert hasattr(c.audio, "transcriptions")
    assert hasattr(c.audio, "translations")
    assert hasattr(c.audio, "speech")
    assert hasattr(c, "moderations")
    assert hasattr(c, "models")
    assert hasattr(c, "completions")
    assert hasattr(c, "message")
    assert hasattr(c, "tools")
    assert hasattr(c, "batch")
    print("  ✅ Client Initialization")

    # Test model dump
    completion = ChatCompletion(
        id="test",
        model="gpt-4",
        choices=[Choice(
            index=0,
            message=ChatMessage(role="assistant", content="Hi"),
            finish_reason="stop"
        )],
        usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6)
    )
    d = completion.model_dump()
    assert d["id"] == "test"
    assert d["choices"][0]["message"]["content"] == "Hi"
    j = completion.model_dump_json()
    parsed = json.loads(j)
    assert parsed["model"] == "gpt-4"
    print("  ✅ Model Serialization")

    # Test response format helpers
    rf_json = ToolDefinition.response_format_json_object()
    assert rf_json["type"] == "json_object"
    rf_schema = ToolDefinition.response_format_json("test", {"type": "object"}, strict=True)
    assert rf_schema["type"] == "json_schema"
    assert rf_schema["json_schema"]["name"] == "test"
    print("  ✅ Response Format Helpers")

    # Test merge_chunks utility
    chunks2 = [
        _parse_chunk({"id": "c1", "model": "m", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}),
        _parse_chunk({"id": "c1", "model": "m", "choices": [{"index": 0, "delta": {"content": "A"}, "finish_reason": None}]}),
        _parse_chunk({"id": "c1", "model": "m", "choices": [{"index": 0, "delta": {"content": "B"}, "finish_reason": "stop"}]}),
    ]
    merged = merge_chunks(chunks2)
    assert merged.choices[0].message.content == "AB"
    print("  ✅ merge_chunks()")

    print()
    print("All internal tests passed! ✅")
    print()
    print(f"Library ready. Import with: from API import Client")