"""Tests verifying header injection, bearer-token auth, interstitial bypass,
SSE think/answer splitting, and the keep-alive heartbeat.

All transport tests run against httpx.MockTransport — no network access.
"""
import asyncio
import json

import httpx
import pytest

from app.config import ALT_USER_AGENT, BYPASS_USER_AGENT, Settings
from core.heartbeat import HeartbeatMonitor
from core.tunnel_client import (
    TunnelAuthError,
    TunnelClient,
    ThinkSplitter,
)

TOKEN = "test-session-token-0123456789abcdef"
BASE = "https://random-words.trycloudflare.com"
SSE_FRAMES = (
    b'data: {"choices":[{"delta":{"content":"<think>step 1: check"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":" volumes"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"</think>Assessment: "}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"volume overload."}}]}\n\n'
    b"data: [DONE]\n\n"
)
_seen: list = []


def make_handler(interstitial_once: bool = True, fail_auth: bool = False):
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        _seen.append(
            {
                "url": str(request.url),
                "ua": request.headers.get("user-agent"),
                "auth": request.headers.get("authorization"),
                "xclient": request.headers.get("x-medharness-client"),
            }
        )
        path = request.url.path
        if fail_auth or request.headers.get("authorization") != f"Bearer {TOKEN}":
            return httpx.Response(401, json={"error": "unauthorized"})
        if path == "/health":
            state["calls"] += 1
            if interstitial_once and state["calls"] == 2:
                return httpx.Response(200, text="<html>Just a moment… challenge</html>",
                                      headers={"content-type": "text/html"})
            return httpx.Response(200, json={"status": "ok"})
        if path == "/v1/chat/completions":
            ua = request.headers.get("user-agent")
            if interstitial_once and ua == BYPASS_USER_AGENT and state.setdefault("blocked_once", False) is False:
                state["blocked_once"] = True
                return httpx.Response(200, text="<html>Attention Required! cf-challenge</html>",
                                      headers={"content-type": "text/html"})
            body = json.loads(request.content or b"{}")
            if body.get("stream"):
                return httpx.Response(200, content=SSE_FRAMES,
                                      headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "Assessment: volume overload."}}]})
        return httpx.Response(404)
    return handler


def make_client(handler) -> TunnelClient:
    settings = Settings(session_token=TOKEN, retry_attempts=2, retry_backoff_s=0.01)
    return TunnelClient(BASE, TOKEN, settings, transport=httpx.MockTransport(handler))


# ------------------------------------------------------------- auth --------
def test_bearer_and_bypass_headers_injected_on_every_request():
    async def run():
        client = make_client(make_handler(interstitial_once=False))
        try:
            await client.health()
            await client.chat("medr1", [{"role": "user", "content": "hi"}])
        finally:
            await client.aclose()
    asyncio.run(run())
    assert len(_seen) >= 2
    for hit in _seen:
        assert hit["auth"] == f"Bearer {TOKEN}"
        assert hit["ua"] == BYPASS_USER_AGENT  # curl-UA bypass on ALL requests
        assert hit["xclient"] == "local-harness/1.0"
    _seen.clear()


def test_rejected_token_raises_auth_error():
    async def run():
        client = make_client(make_handler(fail_auth=True))
        try:
            with pytest.raises(TunnelAuthError):
                await client.health()
        finally:
            await client.aclose()
    asyncio.run(run())
    _seen.clear()


# ------------------------------------------------- interstitial bypass -----
def test_cloudflare_interstitial_bypassed_via_alt_ua():
    """First attempt (curl UA) gets a challenge page → retry with browser UA."""
    async def run():
        client = make_client(make_handler())
        try:
            modes: list = []
            async for mode, text in client.stream_chat(
                "medr1", [{"role": "user", "content": "assess"}]
            ):
                modes.append((mode, text))
            assert client.last_bypass_used == ALT_USER_AGENT
            return modes
        finally:
            await client.aclose()
    modes = asyncio.run(run())
    # think segments and answer segments arrive correctly split
    think = "".join(t for m, t in modes if m == "think")
    answer = "".join(t for m, t in modes if m == "answer")
    assert think == "step 1: check volumes"
    assert answer == "Assessment: volume overload."
    assert _seen[-1]["ua"] == ALT_USER_AGENT
    _seen.clear()


def test_sse_reasoning_content_field_also_routes_to_think():
    """vLLM reasoning models may emit delta.reasoning_content — must map to think."""
    frames = (
        b'data: {"choices":[{"delta":{"reasoning_content":"hop 1"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"final"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=frames, headers={"content-type": "text/event-stream"})

    async def run():
        client = make_client(handler)
        try:
            return [x async for x in client.stream_chat("medr1", [])]
        finally:
            await client.aclose()
    out = asyncio.run(run())
    assert ("think", "hop 1") in out and ("answer", "final") in out


# ------------------------------------------------- think splitter unit -----
def test_think_splitter_handles_tags_split_across_chunks():
    s = ThinkSplitter()
    out: list = []
    for chunk in ["<thi", "nk>de", "ep re", "ason", "ing</", "thin", "k>ans", "wer"]:
        out.extend(s.feed(chunk))
    out.extend(s.flush())
    think = "".join(t for m, t in out if m == "think")
    answer = "".join(t for m, t in out if m == "answer")
    assert think == "deep reasoning"
    assert answer == "answer"


def test_think_splitter_no_tag_passthrough():
    s = ThinkSplitter()
    out = []
    for chunk in "plain answer text with <b>markup</b> but no think tags".split(" "):
        out.extend(s.feed(chunk + " "))
    out.extend(s.flush())
    assert all(m == "answer" for m, _ in out)
    assert "".join(t for _, t in out) == "plain answer text with <b>markup</b> but no think tags "


# ------------------------------------------------------------- heartbeat ---
def test_heartbeat_counts_failures_and_recovers():
    handler = make_handler(interstitial_once=False)
    responses = {"n": 0}
    original = handler

    def flaky(request: httpx.Request) -> httpx.Response:
        responses["n"] += 1
        if responses["n"] in (2, 3, 4):
            return httpx.Response(502)
        return original(request)

    async def run():
        settings = Settings(session_token=TOKEN, retry_attempts=1, retry_backoff_s=0.0)
        client = TunnelClient(BASE, TOKEN, settings, transport=httpx.MockTransport(flaky))
        hb = HeartbeatMonitor(client, interval_s=0.02, jitter_s=0.0, failure_threshold=3)
        hb.start()
        max_failures = 0
        try:
            for _ in range(20):  # sample max mid-flight before full recovery
                await asyncio.sleep(0.02)
                max_failures = max(max_failures, hb.consecutive_failures)
                if hb.last_ok_at is not None and responses["n"] > 4:
                    break
        finally:
            await hb.stop()
            await client.aclose()
        return hb, max_failures

    hb, max_failures = asyncio.run(run())
    assert hb.pings_sent >= 4
    assert max_failures >= 1          # 502s were observed and counted
    assert hb.consecutive_failures == 0  # …then self-healed on later pings
    assert hb.healthy is True and hb.last_ok_at is not None
    assert hb.last_error is None


def test_heartbeat_healthy_path():
    async def run():
        client = make_client(make_handler(interstitial_once=False))
        hb = HeartbeatMonitor(client, interval_s=0.02, jitter_s=0.0)
        hb.start()
        try:
            await asyncio.sleep(0.2)
        finally:
            await hb.stop()
            await client.aclose()
        return hb
    hb = asyncio.run(run())
    assert hb.healthy is True
    assert hb.consecutive_failures == 0
    assert hb.last_ok_at is not None
    _seen.clear()
