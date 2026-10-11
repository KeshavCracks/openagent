"""Groq-driven headless provisioning agent (the "Setup Engine").

Split-brain rule enforced here: **Groq is never used for clinical traffic.**
It acts strictly as an automated infrastructure engineer that proposes
deployment parameters; execution is performed exclusively by the audited,
canonical `core/cloud_bootstrap.sh`. Groq's output is therefore *data*
(parameters), never *code* — which eliminates the entire class of
prompt-injection-into-shell vulnerabilities.

Anti-rationalization gates:
  * If Groq suggests AWQ or FP8 (the Quantization Trap), the proposal is
    REJECTED and a corrective feedback round is issued.
  * Values are clamped into T4-safe ranges regardless of what Groq returns.
  * If Groq is unreachable after retries, the orchestrator degrades
    gracefully to canonical parameters so the doctor is never blocked.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, field_validator

from app.config import (
    BYPASS_USER_AGENT,
    MODEL_REGISTRY,
    Settings,
    ensure_session_token,
)

_BOOTSTRAP_PATH = Path(__file__).with_name("cloud_bootstrap.sh")

_FORBIDDEN_TOKENS = re.compile(r"\b(awq|fp8|fp8_e4m3|awq_marlin)\b", re.IGNORECASE)
_TUNNEL_RE = re.compile(r"^https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com/?$", re.IGNORECASE)
_HF_REPO_RE = re.compile(r"^[\w][\w./-]{0,180}$")

PROVISION_SYSTEM_PROMPT = """You are a headless DevOps engineer embedded in MedHarness.
Your ONLY job: propose deployment parameters for a free NVIDIA T4 (16 GB, Turing CC 7.5)
remote inference node that will serve TWO clinical models simultaneously:

  * medr1    — Med-R1-8B  (clinical reasoning)   ~5.4 GB at GPTQ W4A16 Marlin
  * medgemma — MedGemma-4B (multimodal imaging)  ~2.9 GB at GPTQ W4A16 Marlin

HARD CONSTRAINTS (non-negotiable, violating them = wrong answer):
  1. Quantization must be GPTQ W4A16 (Marlin kernels). NEVER suggest AWQ
     (up to -16% on medical benchmarks for Gemma3/MedGemma) and NEVER FP8
     (T4 has no FP8 tensor cores).
  2. Both models must remain hot in VRAM simultaneously: keep
     max_model_len values modest (medr1 <= 8192, medgemma <= 4096).
  3. Model repo ids MUST point at GPTQ-quantized artifacts (the runtime
     verifies `quantization_config.quant_method == "gptq"` and refuses
     anything else).

Respond with ONLY a JSON object, no prose, matching exactly:
{
  "vllm_version": "<stable vllm release suitable for CUDA 12.1 / T4>",
  "medr1_gptq_repo": "<HF repo id or empty string if unknown>",
  "medgemma_gptq_repo": "<HF repo id or empty string if unknown>",
  "max_model_len_medr1": <int>,
  "max_model_len_medgemma": <int>,
  "notes": "<one sentence: VRAM budget sanity check>"
}"""


class ProvisionParams(BaseModel):
    """Validated, clamped Groq proposal. These are DATA, never shell code."""

    vllm_version: str = Field(default="0.8.5.post1")
    medr1_gptq_repo: str = Field(default="")
    medgemma_gptq_repo: str = Field(default="")
    max_model_len_medr1: int = Field(default=8192)
    max_model_len_medgemma: int = Field(default=4096)
    notes: str = Field(default="")

    @field_validator("vllm_version")
    @classmethod
    def _vllm_version(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[0-9][0-9A-Za-z.\-+]{0,30}$", v):
            return "0.8.5.post1"  # fall back to pinned canonical
        return v

    @field_validator("max_model_len_medr1")
    @classmethod
    def _cap_r1(cls, v: int) -> int:
        return max(2048, min(int(v), 8192))

    @field_validator("max_model_len_medgemma")
    @classmethod
    def _cap_gemma(cls, v: int) -> int:
        return max(1024, min(int(v), 4096))

    def audit(self) -> List[str]:
        """Anti-rationalization audit. Returns a list of rejections (empty = pass)."""
        problems: List[str] = []
        for name in ("medr1_gptq_repo", "medgemma_gptq_repo", "notes"):
            value = getattr(self, name)
            if _FORBIDDEN_TOKENS.search(value or ""):
                problems.append(
                    f"{name} mentions AWQ/FP8 — the Quantization Trap. "
                    "GPTQ W4A16 Marlin is the ONLY sanctioned format."
                )
        for name in ("medr1_gptq_repo", "medgemma_gptq_repo"):
            value = getattr(self, name)
            if value and not _HF_REPO_RE.match(value):
                problems.append(f"{name} is not a plausible HF repo id / local path: {value!r}")
        return problems


class DeploymentPack(BaseModel):
    """Everything the doctor needs: ONE paste, zero terminal knowledge."""

    colab_cell: str
    kaggle_cell: str
    token: str
    params: ProvisionParams
    tunnel_hint: str = "After bootstrap finishes, copy the printed https://*.trycloudflare.com URL into Settings → Cloud Brain."
    checklist: List[str] = Field(default_factory=list)


class Orchestrator:
    """Async provisioning state machine: PROVISIONING -> AWAITING_TUNNEL -> READY."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------- Groq ----
    async def _ask_groq(self, groq_key: str) -> ProvisionParams:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": PROVISION_SYSTEM_PROMPT},
            {"role": "user", "content": "Propose parameters for a Colab T4 deployment now."},
        ]
        last_rejections: List[str] = []
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {groq_key}", "User-Agent": BYPASS_USER_AGENT},
            timeout=30.0,
        ) as client:
            for attempt in range(2):  # original + one corrective round
                try:
                    resp = await client.post(
                        self.settings.groq_endpoint,
                        json={
                            "model": self.settings.groq_model,
                            "messages": messages,
                            "temperature": 0,
                            "response_format": {"type": "json_object"},
                        },
                    )
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    params = ProvisionParams.model_validate(json.loads(content))
                except Exception as exc:  # noqa: BLE001 — network/parse/validation
                    messages.append({"role": "user", "content": f"REJECTED: {exc}. Return corrected JSON only."})
                    continue
                rejections = params.audit()
                if not rejections:
                    return params
                last_rejections = rejections
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "REJECTED: " + " ".join(rejections)
                    + " Regenerate the JSON obeying all hard constraints.",
                })
        raise RuntimeError(
            "Groq proposal failed the anti-rationalization audit after "
            f"2 rounds: {last_rejections or ['unreachable/invalid output']}"
        )

    def _canonical_params(self) -> ProvisionParams:
        """Safe fallback so the doctor is never blocked by Groq downtime."""
        return ProvisionParams(
            vllm_version="0.8.5.post1",
            medr1_gptq_repo=MODEL_REGISTRY["medr1"].fallback_repo,
            medgemma_gptq_repo=MODEL_REGISTRY["medgemma"].fallback_repo,
            max_model_len_medr1=8192,
            max_model_len_medgemma=4096,
            notes="Canonical offline parameters (Groq unavailable or unevaluated).",
        )

    # -------------------------------------------------- deployment pack ----
    def render_pack(self, params: ProvisionParams, token: str) -> DeploymentPack:
        script = _BOOTSTRAP_PATH.read_text(encoding="utf-8")
        encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
        env = (
            f"SESSION_BEARER_TOKEN={token} "
            f"MEDR1_GPTQ_REPO={params.medr1_gptq_repo or MODEL_REGISTRY['medr1'].fallback_repo} "
            f"MEDGEMMA_GPTQ_REPO={params.medgemma_gptq_repo or MODEL_REGISTRY['medgemma'].fallback_repo} "
            f"VLLM_VERSION={params.vllm_version} "
            f"MAX_MODEL_LEN_R1={params.max_model_len_medr1} "
            f"MAX_MODEL_LEN_GEMMA={params.max_model_len_medgemma}"
        )

        def cell(workdir: str) -> str:
            # base64-wrapped: immune to notebook quoting/escaping problems
            return (
                f"# ==== MedHarness Cloud Brain bootstrap — paste as ONE {workdir.split('/')[0].capitalize()} cell ====\n"
                "import base64, pathlib\n"
                f'script = base64.b64decode("{encoded}").decode()\n'
                f'p = pathlib.Path("{workdir}"); p.mkdir(parents=True, exist_ok=True)\n'
                'p.joinpath("medharness_bootstrap.sh").write_text(script)\n'
                "(p / 'medharness_bootstrap.sh').chmod(0o755)\n"
                f"!env {env} bash {workdir}/medharness_bootstrap.sh\n"
            )

        checklist = [
            "Open colab.research.google.com (or kaggle.com/code) and start a GPU T4 session.",
            "Paste the cell below into a single cell and run it. No typing required.",
            "Wait for the 'CLOUD BRAIN — ONLINE' banner (3–8 minutes).",
            "Copy the printed https://….trycloudflare.com URL into MedHarness Settings → Cloud Brain.",
            "The app's 3-minute heartbeat will keep the VM awake while you review charts.",
        ]
        return DeploymentPack(
            colab_cell=cell("/content"),
            kaggle_cell=cell("/kaggle/working"),
            token=token,
            params=params,
            checklist=checklist,
        )

    async def provision(self, groq_key: Optional[str], allow_fallback: bool = True) -> DeploymentPack:
        """Full provisioning flow. Groq proposes; canonical script disposes."""
        token = self.settings.session_token or ensure_session_token()
        params: Optional[ProvisionParams] = None
        if groq_key:
            try:
                params = await asyncio.wait_for(self._ask_groq(groq_key), timeout=90.0)
            except (RuntimeError, asyncio.TimeoutError) as exc:
                if not allow_fallback:
                    raise
                params = self._canonical_params()
                params.notes = f"Groq unavailable ({type(exc).__name__}) — canonical parameters used."
        else:
            params = self._canonical_params()
            params.notes = "No Groq key supplied — canonical parameters used (clinical work never depends on Groq)."
        return self.render_pack(params, token)

    # --------------------------------------------------- tunnel adoption ---
    async def adopt_tunnel(self, url: str) -> Dict[str, Any]:
        """Validate + health-probe a pasted trycloudflare URL, then arm it."""
        url = url.strip().rstrip("/")
        if not _TUNNEL_RE.match(url):
            raise ValueError("URL must be a https://<name>.trycloudflare.com quick tunnel.")
        from core.tunnel_client import TunnelClient  # local import: no cycle

        client = TunnelClient(url, self.settings.session_token, self.settings)
        try:
            health = await client.health()
        finally:
            await client.aclose()
        self.settings.remote_base_url = url
        self.settings.save()
        return {"status": "ready", "tunnel_url": url, "health": health}
