"""Central configuration, token management, and design tokens.

Portability hard constraint: path handling uses ONLY `pathlib.Path`,
environment lookup uses ONLY `os.environ`, background work uses ONLY
`asyncio`. No platform-specific shell quirks anywhere in this module so
the harness can later be frozen with PyInstaller/AppImage/Tauri without
rewrites.

State is persisted under XDG_STATE_HOME (~/.local/state/medical-harness)
with 0600 file permissions. Override the root with MEDHARNESS_HOME.
"""
from __future__ import annotations

import json
import os
import secrets
import stat
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

APP_TITLE = "MedHarness — Clinical AI Workstation"
VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Cloudflare interstitial bypass (free trycloudflare.com tunnels sometimes
# intercept non-browser clients with a challenge page; curl-style UA passes).
# ---------------------------------------------------------------------------
BYPASS_USER_AGENT = "curl/8.0.0"
ALT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
INTERSTITIAL_MARKERS = (
    "just a moment",
    "attention required",
    "checking your browser",
    "cf-challenge",
    "cf-mitigated",
    "enable javascript and cookies",
)

# ---------------------------------------------------------------------------
# The Quantization Trap — locked architecture constraints surfaced in UI + API
# ---------------------------------------------------------------------------
QUANT_LOCK = {
    "quantization": "GPTQ W4A16 + Marlin kernels",
    "forbidden": ["AWQ", "FP8"],
    "reason": (
        "AWQ drops up to 16% accuracy on Gemma3/MedGemma (RMSNorm + activation "
        "channel scaling); the T4 (Turing, CC 7.5) has no native FP8 tensor cores."
    ),
    "target_vram_gb": {"medr1": 5.4, "medgemma": 2.9, "total_weights": 8.3},
}


# ---------------------------------------------------------------------------
# Model registry (remote cloud-brain). Repos are GPTQ W4A16 checkpoints;
# override via environment or Settings so quantized mirrors/local dirs win.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo_env: str          # env var the bootstrap script reads
    fallback_repo: str     # default candidate resolved on the remote side
    kind: str              # "text" | "multimodal"
    vram_gb: float
    port: int              # upstream vLLM port behind the remote gateway
    description: str


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "medr1": ModelSpec(
        alias="medr1",
        repo_env="MEDR1_GPTQ_REPO",
        fallback_repo="iqvia-ai/Med-R1-8B-GPTQ-W4A16",
        kind="text",
        vram_gb=5.4,
        port=8001,
        description="Clinical logic · multi-hop EHR audits · stepwise diagnostics",
    ),
    "medgemma": ModelSpec(
        alias="medgemma",
        repo_env="MEDGEMMA_GPTQ_REPO",
        fallback_repo="google/medgemma-4b-it-gptq-w4a16",
        kind="multimodal",
        vram_gb=2.9,
        port=8002,
        description="CT/MRI · pathology · CXR · PDF→FHIR/JSON structuring",
    ),
}

SLASH_COMMANDS = ("/spec", "/plan", "/build", "/test", "/review", "/constraints")


# ---------------------------------------------------------------------------
# Pure editorial design tokens (4px grid, WCAG AA). Mirrored in styles.css.
# ---------------------------------------------------------------------------
COLORS: Dict[str, str] = {
    "bg_paper": "#FAFAF7",        # page canvas
    "bg_panel": "#FFFFFF",        # cards / bubbles
    "text_ink": "#1F2A33",        # primary text — 13.9:1 on bg_paper
    "accent_clinical": "#0E7C66", # teal accent — 4.6:1 on bg_paper (AA)
    "accent_strong": "#0A5C4D",   # pressed / emphasis — 6.9:1
    "muted_subtle": "#5B6770",    # secondary text — 5.4:1 on bg_paper
    "line_quiet": "#E3E6E2",      # hairlines
    "danger_clinical": "#B3261E", # alerts — 6.4:1 on bg_paper
    "think_backdrop": "#F1F4F2",  # reasoning accordion canvas
}

GRID_PX = 4  # every coordinate/spacing in the UI is a multiple of 4


def state_dir() -> Path:
    """Resolve (and create) the per-user state directory. stdlib only."""
    env = os.environ.get("MEDHARNESS_HOME")
    if env:
        root = Path(env).expanduser()
    else:
        xdg = os.environ.get("XDG_STATE_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "state"
        root = base / "medical-harness"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_private(path: Path, text: str) -> None:
    """Atomically write a file readable only by the current user (0600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # force 0600 regardless of umask
    os.replace(tmp, path)


def ensure_session_token() -> str:
    """Load or generate the SESSION_BEARER_TOKEN guarding the public tunnel.

    Generated locally with `secrets` (CSPRNG), persisted with 0600. The remote
    vLLM gateway validates this bearer on every request, closing the public
    trycloudflare tunnel to unauthorized traffic.
    """
    token_file = state_dir() / "session_token"
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if len(token) >= 32:
            return token
    token = secrets.token_urlsafe(32)
    _write_private(token_file, token)
    return token


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8787
    # --- Groq: headless DevOps agent ONLY (never used for clinical traffic) ---
    groq_endpoint: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""
    # --- Remote cloud brain (trycloudflare quick tunnel) ---
    remote_base_url: str = ""
    session_token: str = ""
    # --- Transport tuning ---
    connect_timeout_s: float = 15.0
    read_timeout_s: float = 600.0
    retry_attempts: int = 3
    retry_backoff_s: float = 1.0
    # --- Keep-alive heartbeat (Colab idle prevention) ---
    heartbeat_interval_s: float = 180.0
    heartbeat_jitter_s: float = 15.0
    # --- Optional sandboxed public grounding ---
    agent_reach_enabled: bool = False
    # --- Chat context packing ---
    context_chunk_chars: int = 1600

    # -- persistence --------------------------------------------------------
    def save(self) -> None:
        payload = {k: v for k, v in asdict(self).items()}
        _write_private(state_dir() / "settings.json", json.dumps(payload, indent=2))

    @classmethod
    def load(cls) -> "Settings":
        base = cls()
        f = state_dir() / "settings.json"
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            for k, v in data.items():
                if hasattr(base, k):
                    setattr(base, k, v)
        if not base.session_token:
            base.session_token = ensure_session_token()
        return base


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Process-wide settings singleton (loaded once, cheap re-save allowed)."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def set_settings(s: Settings) -> None:
    global _settings
    _settings = s
    s.save()


def public_config() -> Dict[str, Any]:
    """Safe, non-secret config for the UI layer."""
    return {
        "app_title": APP_TITLE,
        "version": VERSION,
        "colors": COLORS,
        "grid_px": GRID_PX,
        "quant_lock": QUANT_LOCK,
        "models": {
            alias: {
                "alias": m.alias,
                "kind": m.kind,
                "vram_gb": m.vram_gb,
                "description": m.description,
                "fallback_repo": m.fallback_repo,
            }
            for alias, m in MODEL_REGISTRY.items()
        },
        "slash_commands": SLASH_COMMANDS,
        "heartbeat_interval_s": get_settings().heartbeat_interval_s,
        "agent_reach_enabled": get_settings().agent_reach_enabled,
    }
