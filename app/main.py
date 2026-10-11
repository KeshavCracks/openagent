"""MedHarness local application entrypoint (FastAPI + local web UI).

Run: `python3 -m app.main` (or ./run.sh) → opens http://127.0.0.1:8787.

This server is the SPLIT-BRAIN pivot:
  * local — math interception, context packing, quality gates, UI serving;
  * remote — ALL clinical inference goes through the bearer-authenticated
    tunnel to OUR vLLM gateway; Groq is only touched by /api/deploy.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import (  # noqa: E402
    APP_TITLE,
    MODEL_REGISTRY,
    VERSION,
    get_settings,
    public_config,
    set_settings,
)
from core.heartbeat import HeartbeatMonitor  # noqa: E402
from core.math_engine import MathError, try_compute  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from core.tunnel_client import (  # noqa: E402
    TunnelAuthError,
    TunnelClient,
    TunnelError,
)
from plugins.agent_reach.reach import pubmed_search  # noqa: E402
from plugins.context_optimizer.optimizer import pack  # noqa: E402
from plugins.quality_gates.engine import (  # noqa: E402
    GateBlock,
    handle_slash,
    validate_output,
)

UI_DIR = Path(__file__).resolve().parent / "ui"

SYSTEM_PROMPT = (
    "You are MedHarness, a clinical decision-support assistant for trained "
    "clinicians (not patients). Structure answers as Assessment / "
    "Differential / Plan / Red flags. You are FORBIDDEN from computing "
    "clinical arithmetic (doses, clearances, rates): when a "
    "[DETERMINISTIC MATH] block is provided it is ground truth — cite it, "
    "never recompute. If data is missing, ask for it instead of guessing "
    "numbers. Reason stepwise; keep reasoning inside <think> tags so it "
    "streams into the reasoning accordion."
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=64_000)
    model: str = Field(default="medr1")
    history: List[Dict[str, str]] = Field(default_factory=list, max_length=40)


class SettingsRequest(BaseModel):
    groq_api_key: Optional[str] = None
    remote_base_url: Optional[str] = None
    agent_reach_enabled: Optional[bool] = None
    session_token: Optional[str] = None  # sync with a Colab-notebook-generated token


class TunnelRequest(BaseModel):
    url: str


class DeployRequest(BaseModel):
    groq_api_key: Optional[str] = None


class EvidenceRequest(BaseModel):
    query: str = Field(min_length=2, max_length=400)


app = FastAPI(title=APP_TITLE, version=VERSION, docs_url=None, redoc_url=None)
_state: Dict[str, Any] = {"tunnel": None, "heartbeat": None}


def _tunnel() -> TunnelClient:
    tunnel: Optional[TunnelClient] = _state.get("tunnel")
    if tunnel is None:
        raise HTTPException(409, "Cloud brain not connected — open Settings → Cloud Brain, deploy, and paste the tunnel URL.")
    return tunnel


@app.on_event("startup")
async def _startup() -> None:
    cfg = get_settings()
    if cfg.remote_base_url:
        client = TunnelClient(cfg.remote_base_url, cfg.session_token, cfg)
        _state["tunnel"] = client
        hb = HeartbeatMonitor(client, cfg.heartbeat_interval_s, cfg.heartbeat_jitter_s)
        hb.start()
        _state["heartbeat"] = hb


@app.on_event("shutdown")
async def _shutdown() -> None:
    hb: Optional[HeartbeatMonitor] = _state.get("heartbeat")
    if hb:
        await hb.stop()
    tunnel: Optional[TunnelClient] = _state.get("tunnel")
    if tunnel:
        await tunnel.aclose()


# ------------------------------------------------------------- static UI ---
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/styles.css")
async def styles() -> FileResponse:
    return FileResponse(UI_DIR / "styles.css", media_type="text/css")


@app.get("/components/clinical_pathway.svg")
async def pathway_sample() -> FileResponse:
    return FileResponse(UI_DIR / "components" / "clinical_pathway.svg", media_type="image/svg+xml")


# --------------------------------------------------------------- config ----
@app.get("/api/config")
async def api_config() -> Dict[str, Any]:
    return public_config()


@app.get("/api/status")
async def api_status() -> Dict[str, Any]:
    cfg = get_settings()
    hb: Optional[HeartbeatMonitor] = _state.get("heartbeat")
    return {
        "version": VERSION,
        "tunnel_configured": bool(cfg.remote_base_url),
        "tunnel_url": cfg.remote_base_url,
        "groq_configured": bool(cfg.groq_api_key),
        "heartbeat": hb.status() if hb else {"running": False},
        "agent_reach_enabled": cfg.agent_reach_enabled,
    }


@app.post("/api/settings")
async def api_settings(req: SettingsRequest) -> Dict[str, Any]:
    cfg = get_settings()
    if req.groq_api_key is not None:
        cfg.groq_api_key = req.groq_api_key.strip()
    if req.agent_reach_enabled is not None:
        cfg.agent_reach_enabled = bool(req.agent_reach_enabled)
    if req.session_token is not None:
        token = req.session_token.strip()
        if len(token) < 32:
            raise HTTPException(400, "Session token looks too short — paste the full token printed by the Colab notebook.")
        cfg.session_token = token
    set_settings(cfg)
    if req.remote_base_url:
        await api_tunnel(TunnelRequest(url=req.remote_base_url))
    return {"ok": True}


# ------------------------------------------------------------ deployment ---
@app.post("/api/deploy")
async def api_deploy(req: DeployRequest) -> Dict[str, Any]:
    """Groq-driven headless provisioning. Returns the ONE-paste cell."""
    cfg = get_settings()
    key = (req.groq_api_key or cfg.groq_api_key).strip() or None
    orchestrator = Orchestrator(cfg)
    try:
        pack_result = await orchestrator.provision(key)
    except RuntimeError as exc:
        raise HTTPException(502, f"Provisioning failed the validation gates: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — surface network problems cleanly
        raise HTTPException(502, f"Provisioning error: {type(exc).__name__}: {exc}") from exc
    if key:
        cfg.groq_api_key = key
        set_settings(cfg)
    return pack_result.model_dump()


@app.post("/api/tunnel")
async def api_tunnel(req: TunnelRequest) -> Dict[str, Any]:
    """Adopt a pasted trycloudflare URL: probe, persist, start heartbeat."""
    orchestrator = Orchestrator(get_settings())
    try:
        result = await orchestrator.adopt_tunnel(req.url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except TunnelError as exc:
        raise HTTPException(502, f"Tunnel reachable but unhealthy: {exc}") from exc
    cfg = get_settings()
    old: Optional[TunnelClient] = _state.get("tunnel")
    if old:
        await old.aclose()
    client = TunnelClient(cfg.remote_base_url, cfg.session_token, cfg)
    _state["tunnel"] = client
    hb = HeartbeatMonitor(client, cfg.heartbeat_interval_s, cfg.heartbeat_jitter_s)
    hb.start()
    _state["heartbeat"] = hb
    return result


# ------------------------------------------------------- models & evidence ---
_MODELS_TTL_S = 60.0
_REPO_HINTS = {
    "medr1": ("med-r1", "medr1"),
    "medgemma": ("medgemma",),
}


@app.get("/api/models")
async def api_models() -> Dict[str, Any]:
    """Model health panel: which registered engines are actually served remotely.

    Cached for 60 s so the UI chip can poll without hammering the tunnel.
    Never raises — degraded panels must not break the chat surface.
    """
    cfg = get_settings()
    tunnel: Optional[TunnelClient] = _state.get("tunnel")
    if not cfg.remote_base_url or tunnel is None:
        return {"tunnel_configured": False, "models": [], "warning": None}
    cached = _state.get("models_cache")
    if cached and (time.monotonic() - cached[0]) < _MODELS_TTL_S:
        return cached[1]
    models: List[Dict[str, Any]] = []
    warning: Optional[str] = None
    try:
        raw = await tunnel.list_models()
        remote_ids = [str(m.get("id", "")) for m in raw.get("data", [])]
        for alias, spec in MODEL_REGISTRY.items():
            served = next((rid for rid in remote_ids if any(h in rid.lower() for h in _REPO_HINTS[alias])), None)
            models.append({
                "alias": alias,
                "description": spec.description,
                "kind": spec.kind,
                "served": served is not None,
                "remote_id": served or "",
            })
    except Exception as exc:  # noqa: BLE001 — panel is advisory, chat stays primary
        warning = f"{type(exc).__name__}: {exc}"
    payload = {"tunnel_configured": True, "models": models, "warning": warning}
    _state["models_cache"] = (time.monotonic(), payload)
    return payload


@app.post("/api/evidence")
async def api_evidence(req: EvidenceRequest) -> Dict[str, Any]:
    """Opt-in public grounding: PubMed E-utilities lookup of the query terms.

    Only the query string leaves the machine — never session data or chat
    history (enforced by the shape of agent_reach and its host allowlist).
    """
    cfg = get_settings()
    if not cfg.agent_reach_enabled:
        raise HTTPException(403, "Public grounding is off — enable it in Settings → Modules.")
    out = await asyncio.to_thread(pubmed_search, req.query.strip(), 5)
    results = [
        {"pmid": r.get("pmid", ""), "title": r.get("title", ""),
         "journal": r.get("journal", ""), "year": r.get("year", ""),
         "url": f"https://pubmed.ncbi.nlm.nih.gov/{r.get('pmid', '')}/"}
        for r in out.get("results", [])
    ]
    return {"query": req.query.strip(), "source": "PubMed (NLM E-utilities, read-only)",
            "results": results, "warning": out.get("warning")}


# ------------------------------------------------------------------ chat ---
def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _chat_stream(req: ChatRequest) -> AsyncIterator[str]:
    cfg = get_settings()
    yield _sse("meta", {"model": req.model, "version": VERSION})

    # 1) Local context packing (aggressive; TTFT wins happen client-side)
    packed = pack(req.message, cfg.context_chunk_chars)
    yield _sse("ctx", packed.stats())

    # 2) Slash commands / progressive disclosure
    guideline: Optional[str] = None
    command, guideline_text, local_reply = handle_slash(req.message)
    if local_reply is not None:
        yield _sse("answer", local_reply)
        yield _sse("done", {"answer": local_reply, "local_only": True})
        return
    if command:
        guideline = guideline_text
        yield _sse("gate", {"command": command, "loaded": "guideline (progressive disclosure L1)"})

    # 3) Deterministic math interception (LLM NEVER computes arithmetic)
    math_block: Optional[str] = None
    try:
        math_result = try_compute(req.message)
    except MathError as exc:
        yield _sse("math", {"status": "incomplete", "message": str(exc)})
        math_result = None
    if math_result is not None:
        math_block = math_result.as_context_block()
        yield _sse("math", {"status": "verified", **{
            "value": math_result.value, "unit": math_result.unit,
            "formula": math_result.formula, "interpretation": math_result.interpretation,
            "warnings": math_result.warnings,
        }})

    # 4) Build messages (packed history + ground-truth math block)
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if guideline:
        messages.append({"role": "system", "content": f"Active workflow guideline ({command}):\n{guideline}"})
    if math_block:
        messages.append({"role": "system", "content": math_block})
    for turn in req.history[-8:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            if len(content) > 2400:
                content = pack(content, cfg.context_chunk_chars).text or content
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": packed.text or req.message})

    # 5) Stream from the remote brain over the authenticated tunnel
    answer_parts: List[str] = []
    try:
        tunnel = _tunnel()
        async for mode, text in tunnel.stream_chat(req.model, messages, max_tokens=1536):
            answer_parts.append(text)
            yield _sse(mode, text)
    except TunnelAuthError as exc:
        yield _sse("error", {"kind": "auth", "message": str(exc)})
        return
    except GateBlock as block:  # defensive: gates raise before/after streaming
        yield _sse("gate", {"blocked": True, "gate": block.gate, "reason": block.reason, "details": block.details})
        return
    except (TunnelError, HTTPException) as exc:
        yield _sse("error", {"kind": "transport", "message": str(exc)})
        return

    # 6) Post-generation anti-rationalization audit
    answer = "".join(answer_parts)
    try:
        block = validate_output(answer)
    except Exception:  # noqa: BLE001 — heuristics must never kill the stream
        block = None
    if block is not None:
        yield _sse("gate", {"blocked": True, "gate": block.gate, "reason": block.reason, "details": block.details})
    yield _sse("done", {"answer": answer, "math": bool(math_block), "ctx": packed.stats()})


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> StreamingResponse:
    if req.model not in MODEL_REGISTRY:
        raise HTTPException(400, f"Unknown model alias '{req.model}'.")
    return StreamingResponse(
        _chat_stream(req), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


def main() -> None:
    import uvicorn

    cfg = get_settings()
    url = f"http://{cfg.host}:{cfg.port}"
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
