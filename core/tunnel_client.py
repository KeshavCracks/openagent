"""Authenticated tunnel transport to the remote cloud brain.

Responsibilities (all local-side, stdlib-path rules respected):
  * Persistent HTTP/2 connection pool (graceful HTTP/1.1 fallback when the
    optional `h2` extra is absent).
  * Cloudflare interstitial bypass: every request is sent with
    `User-Agent: curl/8.0.0`; if a challenge page is detected, exactly one
    retry with a browser UA is performed before declaring the tunnel blocked.
  * Bearer-token authentication on every request (SESSION_BEARER_TOKEN).
  * SSE streaming of OpenAI-compatible chat completions, including
    reasoning-model `<think>` segments split statefully into
    ("think", text) / ("answer", text) tuples so the UI can render the
    collapsible reasoning accordion with sub-1.5 s perceived latency.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from app.config import (
    ALT_USER_AGENT,
    BYPASS_USER_AGENT,
    INTERSTITIAL_MARKERS,
    Settings,
)


class TunnelError(Exception):
    """Base transport error."""


class TunnelAuthError(TunnelError):
    """Remote rejected the bearer token — regenerate & redeploy."""


class TunnelBlocked(TunnelError):
    """Cloudflare interstitial persisted after bypass attempts."""


class TransientTunnelError(TunnelError):
    """Retryable network/5xx condition."""


class ThinkSplitter:
    """Stateful splitter for reasoning streams: <think>…</think> segments.

    Handles tags split across chunk boundaries by holding back a tail window
    of (tag_length - 1) characters.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.state: str = "answer"
        self.buf: str = ""

    def _emit_safe(self, mode: str, tag: str, out: List[Tuple[str, str]]) -> None:
        keep = max(0, len(self.buf) - (len(tag) - 1))
        if keep > 0:
            out.append((mode, self.buf[:keep]))
            self.buf = self.buf[keep:]

    def feed(self, chunk: str) -> List[Tuple[str, str]]:
        self.buf += chunk
        out: List[Tuple[str, str]] = []
        while True:
            tag = self.CLOSE if self.state == "think" else self.OPEN
            idx = self.buf.find(tag)
            if idx == -1:
                self._emit_safe(self.state, tag, out)
                break
            if idx > 0:
                out.append((self.state, self.buf[:idx]))
            self.buf = self.buf[idx + len(tag):]
            self.state = "think" if self.state == "answer" else "answer"
            if not self.buf:
                break
        return out

    def flush(self) -> List[Tuple[str, str]]:
        if not self.buf:
            return []
        out = [(self.state, self.buf)]
        self.buf = ""
        return out


class TunnelClient:
    """Async client for the remote MedHarness gateway (OpenAI-compatible)."""

    def __init__(
        self,
        base_url: str,
        token: str,
        settings: Optional[Settings] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.settings = settings or Settings()
        timeout = httpx.Timeout(
            self.settings.read_timeout_s,
            connect=self.settings.connect_timeout_s,
            pool=30.0,
        )
        try:
            self._client = httpx.AsyncClient(
                http2=True, transport=transport, timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0),
            )
        except ImportError:  # h2 not installed -> HTTP/1.1 fallback
            self._client = httpx.AsyncClient(
                transport=transport, timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0),
            )
        self.last_bypass_used: Optional[str] = None

    # ------------------------------------------------------------ headers --
    def _headers(self, user_agent: str = BYPASS_USER_AGENT, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": user_agent,             # Cloudflare interstitial bypass
            "Accept": "application/json",
            "X-MedHarness-Client": "local-harness/1.0",
        }
        if extra:
            headers.update(extra)
        return headers

    @staticmethod
    def _looks_interstitial(content_type: str, body_head: str, headers: Optional[httpx.Headers] = None) -> bool:
        if headers and "challenge" in (headers.get("cf-mitigated") or "").lower():
            return True
        if "text/html" in (content_type or "").lower():
            lowered = body_head.lower()
            return any(marker in lowered for marker in INTERSTITIAL_MARKERS)
        return False

    # ------------------------------------------------------------ health ---
    async def health(self) -> Dict[str, Any]:
        return await self._request_json("GET", f"{self.base_url}/health")

    async def list_models(self) -> Dict[str, Any]:
        return await self._request_json("GET", f"{self.base_url}/v1/models")

    async def _request_json(self, method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        delay = self.settings.retry_backoff_s
        last_error: Optional[Exception] = None
        for attempt in range(max(1, self.settings.retry_attempts)):
            try:
                resp = await self._client.request(method, url, json=payload, headers=self._headers())
                if resp.status_code in (401, 403):
                    raise TunnelAuthError("Remote rejected the session token — regenerate & redeploy.")
                if resp.status_code >= 500:
                    raise TransientTunnelError(f"Remote {resp.status_code}")
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                text = resp.text
                if self._looks_interstitial(ctype, text[:2048], resp.headers):
                    raise TunnelBlocked("Cloudflare interstitial detected.")
                self.last_bypass_used = BYPASS_USER_AGENT
                return resp.json()
            except (httpx.TransportError, httpx.HTTPStatusError, TransientTunnelError) as exc:
                last_error = exc
                if attempt < self.settings.retry_attempts - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
        raise TransientTunnelError(f"{type(last_error).__name__}: {last_error}")

    # ------------------------------------------------------------- chat ----
    async def chat(
        self, model_alias: str, messages: List[Dict[str, str]],
        *, temperature: float = 0.2, max_tokens: int = 1024, extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Non-streaming completion (used by tools/tests; UI uses stream_chat)."""
        payload: Dict[str, Any] = {
            "model": model_alias, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "stream": False,
        }
        if extra:
            payload.update(extra)
        data = await self._request_json("POST", f"{self.base_url}/v1/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    async def stream_chat(
        self, model_alias: str, messages: List[Dict[str, str]],
        *, temperature: float = 0.2, max_tokens: int = 1024, extra: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Tuple[str, str]]:
        """Yield ("think"|"answer", delta_text) tuples over SSE."""
        payload: Dict[str, Any] = {
            "model": model_alias, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "stream": True,
        }
        if extra:
            payload.update(extra)
        url = f"{self.base_url}/v1/chat/completions"
        attempts = (BYPASS_USER_AGENT, ALT_USER_AGENT)  # interstitial bypass ladder
        emitted = False
        last_error: Optional[Exception] = None
        for ua in attempts:
            splitter = ThinkSplitter()
            try:
                async with self._client.stream("POST", url, json=payload, headers=self._headers(ua)) as resp:
                    if resp.status_code in (401, 403):
                        raise TunnelAuthError("Remote rejected the session token — regenerate & redeploy.")
                    if resp.status_code >= 500:
                        raise TransientTunnelError(f"Remote {resp.status_code}")
                    ctype = resp.headers.get("content-type", "")
                    head = b""
                    if "text/html" in ctype.lower():
                        async for chunk in resp.aiter_bytes():
                            head += chunk
                            if len(head) > 2048:
                                break
                        if self._looks_interstitial(ctype, head.decode("utf-8", "ignore"), resp.headers):
                            raise TunnelBlocked("Cloudflare interstitial — retrying with browser UA.")
                        raise TunnelBlocked("Unexpected HTML payload from tunnel.")
                    self.last_bypass_used = ua
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        reasoning = delta.get("reasoning_content")
                        content = delta.get("content")
                        if reasoning:
                            emitted = True
                            yield "think", reasoning
                        if content:
                            for mode, piece in splitter.feed(content):
                                if piece:
                                    emitted = True
                                    yield mode, piece
                    for mode, piece in splitter.flush():
                        if piece:
                            emitted = True
                            yield mode, piece
                return
            except (TunnelAuthError,):
                raise
            except (httpx.TransportError, TransientTunnelError, TunnelBlocked) as exc:
                last_error = exc
                if emitted:
                    raise  # never re-stream mid-answer
                continue
        raise TunnelError(f"Stream failed after bypass attempts: {last_error}")

    async def aclose(self) -> None:
        await self._client.aclose()
