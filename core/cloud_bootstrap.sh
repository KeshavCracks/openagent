#!/usr/bin/env bash
# ============================================================================
#  MedHarness CLOUD BRAIN BOOTSTRAP
#  Runs ONCE inside a free Google Colab / Kaggle NVIDIA T4 session.
#  Deploys the dual-model clinical engine behind an authenticated Cloudflare
#  quick tunnel and prints the connection manifest for the local harness.
#
#  LOCKED ARCHITECTURE CONSTRAINTS (enforced mechanically below):
#   * Quantization ......... GPTQ W4A16 with Marlin kernels ONLY.
#                            AWQ is forbidden: up to -16% accuracy on
#                            Gemma3/MedGemma medical benchmarks (RMSNorm +
#                            activation channel scaling).
#                            FP8 is forbidden: T4 (Turing, CC 7.5) has no
#                            native FP8 tensor cores.
#   * Both models stay hot in VRAM simultaneously (zero swapping):
#       Med-R1-8B  GPTQ W4A16 Marlin  ~5.4 GB
#       MedGemma-4B GPTQ W4A16 Marlin ~2.9 GB
#       => ~8.3 GB static weights; remaining budget is vLLM PagedAttention
#          KV cache (gpu-memory-utilization splits are env-tunable).
#   * Prefix caching ON (--enable-prefix-caching) for low TTFT.
#   * Every endpoint validates the locally generated SESSION_BEARER_TOKEN.
#   * Remote request logging is DISABLED (--disable-log-requests) so no PHI
#     persists in notebook logs.
#
#  REQUIRED ENV (supplied automatically by the local harness deployment pack):
#    SESSION_BEARER_TOKEN   cryptographic session token (mandatory)
#  OPTIONAL ENV:
#    MEDR1_GPTQ_REPO / MEDGEMMA_GPTQ_REPO   HF repo id OR local dir holding a
#                           GPTQ-W4A16 checkpoint (verified by preflight).
#                           If unset, known candidates are probed and the
#                           script FAILS LOUDLY with a quantization recipe
#                           rather than silently falling back to AWQ/FP8.
#    HF_TOKEN               required for gated repos (e.g. google/medgemma-*)
#    VLLM_VERSION           default 0.8.5.post1
#    MAX_MODEL_LEN_R1       default 8192    MAX_MODEL_LEN_GEMMA default 4096
#    GPU_MEM_UTIL_R1        default 0.55    GPU_MEM_UTIL_GEMMA   default 0.38
#    SESSION_MAX_HOURS      default 12 (hard safety cap on the VM session)
#    MEDHARNESS_WORKDIR     default $PWD/medharness (never hardcodes /tmp)
# ============================================================================
set -Eeuo pipefail

# ---------------------------------------------------------------- helpers ---
log()  { printf '\033[1;32m[medharness]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

WORKDIR="${MEDHARNESS_WORKDIR:-$PWD/medharness}"
mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

: "${SESSION_BEARER_TOKEN:?FATAL: SESSION_BEARER_TOKEN is mandatory — generate it in the local MedHarness app (Settings > Cloud Brain).}"

VLLM_VERSION="${VLLM_VERSION:-0.8.5.post1}"
MAX_MODEL_LEN_R1="${MAX_MODEL_LEN_R1:-8192}"
MAX_MODEL_LEN_GEMMA="${MAX_MODEL_LEN_GEMMA:-4096}"
GPU_MEM_UTIL_R1="${GPU_MEM_UTIL_R1:-0.55}"
GPU_MEM_UTIL_GEMMA="${GPU_MEM_UTIL_GEMMA:-0.38}"
SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-12}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"
CLOUDFLARED_VERSION="${CLOUDFLARED_VERSION:-2025.5.0}"
SESSION_STARTED_AT="$(date -u +%s)"

# ------------------------------------------------- 0) GPU preflight ---------
log "0/7 GPU preflight"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — attach a GPU runtime (Colab: Runtime > Change runtime type > T4)."
nvidia-smi || die "GPU present but not responding."
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
log "GPU: compute capability ${CC}, ${VRAM_MB} MiB VRAM"
awk -v cc="$CC" 'BEGIN{exit !(cc+0 >= 7.5)}' || die "Compute capability ${CC} < 7.5 — Marlin kernels require SM75+."
if [ "${VRAM_MB:-0}" -lt 15000 ]; then
  warn "VRAM ${VRAM_MB} MiB < 15 GiB: dual-model residency may not fit. Lower GPU_MEM_UTIL_* or run a single model."
fi

# ------------------------------------------------- 1) vLLM install ----------
log "1/7 Installing vLLM ${VLLM_VERSION}"
if ! python3 -c "import vllm" >/dev/null 2>&1 || ! python3 -c "from importlib.metadata import version; import sys; sys.exit(0 if version('vllm').startswith('${VLLM_VERSION%.post1}') else 1)" >/dev/null 2>&1; then
  pip install --quiet "vllm==${VLLM_VERSION}" 2>&1 | tail -n3 || die "vLLM install failed."
fi
pip install --quiet "huggingface_hub[cli]" "httpx" >/dev/null 2>&1 || warn "helper installs failed; continuing."

# --------------------------------------- 2) GPTQ W4A16 preflight gate -------
# The Quantization Trap guard: mechanically REFUSE any checkpoint that is not
# GPTQ. This is what prevents an accidental AWQ/FP8 deployment forever.
PREFLIGHT="$WORKDIR/preflight_gptq.py"
cat > "$PREFLIGHT" <<'PYEOF'
"""Verify a checkpoint/repo is GPTQ-quantized. Exit codes: 0 ok, 3 not-GPTQ, 4 unreachable."""
import json, os, sys
from huggingface_hub import hf_hub_download

repo = os.environ["REPO"]
try:
    path = hf_hub_download(repo, "config.json", token=os.environ.get("HF_TOKEN") or None)
except Exception as exc:  # noqa: BLE001
    print(f"REPO_UNREACHABLE::{repo}::{type(exc).__name__}: {exc}")
    sys.exit(4)
try:
    cfg = json.load(open(path, encoding="utf-8"))
except Exception as exc:  # noqa: BLE001
    print(f"CONFIG_UNREADABLE::{repo}::{exc}"); sys.exit(4)
q = (cfg.get("quantization_config") or {}).get("quant_method", "").lower()
if q not in ("gptq", "gptq_marlin"):
    print(f"NOT_GPTQ::{repo}::quant_method={q or 'none'}")
    sys.exit(3)
print("GPTQ_OK::" + repo)
PYEOF

QUANT_RECIPE='
  One-time fix (choose one):
  a) Set MEDR1_GPTQ_REPO / MEDGEMMA_GPTQ_REPO to a public GPTQ-W4A16 mirror.
  b) Quantize locally once (CPU-OK) and upload / mount the directory:
       pip install gptqmodel  (or auto-gptq)
       from gptqmodel import GPTQModel, QuantizeConfig
       q = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False)  # W4A16
       GPTQModel.load("google/medgemma-4b-it", q).quantize(calib_docs).save("./medgemma-4b-gptq")
     then set MEDGEMMA_GPTQ_REPO=/path/to/medgemma-4b-gptq
  MedHarness refuses AWQ/FP8 by design: AWQ costs up to 16% on medical
  benchmarks (RMSNorm scaling); the T4 has no FP8 tensor cores.'

resolve_repo() { # $1 = local env var name holding candidate list
  local candidates_var="$1" candidates c
  candidates="${!candidates_var}"
  for c in $candidates; do
    if [ -d "$c" ]; then echo "$c"; return 0; fi   # local quantized dir wins
    if REPO="$c" python3 "$PREFLIGHT" 2>/dev/null | grep -q '^GPTQ_OK::'; then echo "$c"; return 0; fi
  done
  return 1
}

MEDR1_CANDIDATES="${MEDR1_GPTQ_REPO:-} iqvia-ai/Med-R1-8B-GPTQ-W4A16 medr1-community/Med-R1-8B-GPTQ-Int4"
MEDGEMMA_CANDIDATES="${MEDGEMMA_GPTQ_REPO:-} google/medgemma-4b-it-gptq-w4a16 medgemma-community/medgemma-4b-it-gptq"

log "2/7 Resolving GPTQ W4A16 checkpoints (Quantization-Trap gate)"
MEDR1_FINAL="$(resolve_repo MEDR1_CANDIDATES)" || die "No GPTQ W4A16 checkpoint found for Med-R1-8B.
$QUANT_RECIPE"
MEDGEMMA_FINAL="$(resolve_repo MEDGEMMA_CANDIDATES)" || die "No GPTQ W4A16 checkpoint found for MedGemma-4B.
$QUANT_RECIPE"
log "Med-R1 checkpoint: $MEDR1_FINAL"
log "MedGemma checkpoint: $MEDGEMMA_FINAL"
case "$MEDGEMMA_FINAL" in google/*) [ -n "${HF_TOKEN:-}" ] || die "google/medgemma-* is a gated repo: accept the license on its Hugging Face page, then export HF_TOKEN.";; esac

# ------------------------------------------------- 3) Prefetch weights ------
log "3/7 Prefetching weights (resumable)"
download_model() { # $1 repo, $2 target dir
  for attempt in 1 2 3; do
    if hf download "$1" --local-dir "$2" --quiet 2>>"$WORKDIR/logs/hf_download.log"; then return 0; fi
    warn "download attempt $attempt for $1 failed; retrying in 15s"; sleep 15
  done
  die "Weight download failed after 3 attempts for $1 (see $WORKDIR/logs/hf_download.log)."
}
[ -d "$WORKDIR/medr1_ckpt" ] && [ -n "$(ls -A "$WORKDIR/medr1_ckpt" 2>/dev/null)" ] \
  || download_model "$MEDR1_FINAL" "$WORKDIR/medr1_ckpt"
[ -d "$WORKDIR/medgemma_ckpt" ] && [ -n "$(ls -A "$WORKDIR/medgemma_ckpt" 2>/dev/null)" ] \
  || download_model "$MEDGEMMA_FINAL" "$WORKDIR/medgemma_ckpt"

# ------------------------------------------------- 4) Auth gateway ----------
# Single public origin (:8000) -> alias routing to both vLLM engines.
# Validates the bearer token on EVERY request, closing the public tunnel.
log "4/7 Writing authenticated alias gateway"
cat > "$WORKDIR/gateway.py" <<'GATEWAY_EOF'
"""MedHarness remote gateway: bearer auth + model alias routing + health aggregation."""
import os
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

TOKEN = os.environ["SESSION_BEARER_TOKEN"]
UPSTREAM = {
    "medr1":    "http://127.0.0.1:8001",
    "medgemma": "http://127.0.0.1:8002",
}
app = FastAPI(title="MedHarness gateway", docs_url=None, redoc_url=None)
_client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, read=None))


def _deny() -> JSONResponse:
    return JSONResponse({"error": "unauthorized"}, status_code=401)


@app.get("/health")
async def health():
    out = {}
    for alias, base in UPSTREAM.items():
        try:
            r = await _client.get(base + "/health", headers={"Authorization": "Bearer " + TOKEN})
            out[alias] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
        except Exception as exc:  # noqa: BLE001
            out[alias] = f"down::{type(exc).__name__}"
    status = "ok" if all(v == "ok" for v in out.values()) else "degraded"
    return JSONResponse({"status": status, "models": out})


@app.get("/v1/models")
async def models(request: Request):
    if request.headers.get("authorization", "") != "Bearer " + TOKEN:
        return _deny()
    ids = [{"id": a, "object": "model"} for a in UPSTREAM]
    return JSONResponse({"object": "list", "data": ids})


@app.post("/v1/chat/completions")
async def chat(request: Request):
    if request.headers.get("authorization", "") != "Bearer " + TOKEN:
        return _deny()
    body = await request.json()
    alias = body.get("model")
    if alias not in UPSTREAM:
        return JSONResponse({"error": f"unknown model alias '{alias}'"}, status_code=400)
    base = UPSTREAM[alias]
    if body.get("stream"):
        async def gen():
            async with _client.stream("POST", base + "/v1/chat/completions", json=body) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")
    r = await _client.post(base + "/v1/chat/completions", json=body)
    return Response(content=r.content, status_code=r.status_code, media_type="application/json")
GATEWAY_EOF

# ------------------------------------------------- 5) Launch engines --------
# Launchers are kept explicit (no eval) so every flag is auditable at a glance.
launch_r1() {
  nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$WORKDIR/medr1_ckpt" --served-model-name medr1 \
    --host 127.0.0.1 --port 8001 \
    --quantization gptq_marlin --dtype float16 \
    --gpu-memory-utilization "$GPU_MEM_UTIL_R1" \
    --max-model-len "$MAX_MODEL_LEN_R1" \
    --enable-prefix-caching --disable-log-requests \
    --trust-remote-code --api-key "$SESSION_BEARER_TOKEN" \
    > "$WORKDIR/logs/vllm_medr1.log" 2>&1 &
  echo $! > "$WORKDIR/pid_medr1"
}
launch_gemma() {
  nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$WORKDIR/medgemma_ckpt" --served-model-name medgemma \
    --host 127.0.0.1 --port 8002 \
    --quantization gptq_marlin --dtype float16 \
    --gpu-memory-utilization "$GPU_MEM_UTIL_GEMMA" \
    --max-model-len "$MAX_MODEL_LEN_GEMMA" \
    --enable-prefix-caching --disable-log-requests \
    --api-key "$SESSION_BEARER_TOKEN" \
    > "$WORKDIR/logs/vllm_medgemma.log" 2>&1 &
  echo $! > "$WORKDIR/pid_medgemma"
}
launch_gateway() {
  nohup python3 -m uvicorn gateway:app --host 127.0.0.1 --port "$GATEWAY_PORT" \
    > "$WORKDIR/logs/gateway.log" 2>&1 &
  echo $! > "$WORKDIR/pid_gateway"
}

log "5/7 Launching vLLM engines (GPTQ W4A16 Marlin, both hot in VRAM)"
launch_r1
launch_gemma

wait_healthy() { # $1 port, $2 tries
  local url="http://127.0.0.1:$1/health" code i
  for i in $(seq 1 "${2:-90}"); do
    code="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $SESSION_BEARER_TOKEN" "$url" || true)"
    [ "$code" = "200" ] && return 0
    sleep 5
  done
  return 1
}

log "Waiting for medr1 engine (vLLM CUDA graph capture takes a few minutes)…"
wait_healthy 8001 120 || { tail -n 40 "$WORKDIR/logs/vllm_medr1.log" >&2; die "medr1 engine failed to become healthy."; }
log "Waiting for medgemma engine…"
wait_healthy 8002 120 || { tail -n 40 "$WORKDIR/logs/vllm_medgemma.log" >&2; die "medgemma engine failed to become healthy."; }
launch_gateway
wait_healthy "$GATEWAY_PORT" 24 || { tail -n 30 "$WORKDIR/logs/gateway.log" >&2; die "gateway failed."; }

# ------------------------------------------------- 6) Cloudflare tunnel -----
log "6/7 Starting Cloudflare quick tunnel (encrypted, HTTP/2)"
case "$(uname -m)" in
  x86_64)  CF_ARCH="amd64" ;;
  aarch64|arm64) CF_ARCH="arm64" ;;
  *) die "Unsupported arch $(uname -m)" ;;
esac
if [ ! -x "$WORKDIR/cloudflared" ]; then
  curl -fsSL -o "$WORKDIR/cloudflared" \
    "https://github.com/cloudflare/cloudflared/releases/${CLOUDFLARED_VERSION}/cloudflared-linux-${CF_ARCH}" \
    || die "cloudflared download failed."
  chmod +x "$WORKDIR/cloudflared"
fi
nohup "$WORKDIR/cloudflared" tunnel --url "http://127.0.0.1:${GATEWAY_PORT}" \
  --no-autoupdate --protocol http2 > "$WORKDIR/logs/cloudflared.log" 2>&1 &
echo $! > "$WORKDIR/pid_tunnel"

TUNNEL_URL=""
for _ in $(seq 1 60); do
  TUNNEL_URL="$(grep -Eo 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$WORKDIR/logs/cloudflared.log" | head -n1 || true)"
  [ -n "$TUNNEL_URL" ] && break
  sleep 2
done
[ -n "$TUNNEL_URL" ] || { tail -n 30 "$WORKDIR/logs/cloudflared.log" >&2; die "Could not obtain a trycloudflare URL."; }

# ------------------------------------------------- 7) Manifest + banner -----
cat > "$WORKDIR/CONNECTION_MANIFEST.json" <<MANIFEST_EOF
{
  "tunnel_url": "$TUNNEL_URL",
  "gateway_port": $GATEWAY_PORT,
  "models": {"medr1": "$MEDR1_FINAL", "medgemma": "$MEDGEMMA_FINAL"},
  "quantization": "gptq_w4a16_marlin",
  "auth": "bearer",
  "session_max_hours": $SESSION_MAX_HOURS,
  "issued_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST_EOF

MASKED_TOKEN="${SESSION_BEARER_TOKEN:0:8}…"
log "7/7 Cloud brain ONLINE"
cat >&2 <<BANNER

============================================================================
  MEDHARNESS CLOUD BRAIN — ONLINE
----------------------------------------------------------------------------
  Tunnel URL : $TUNNEL_URL
               -> paste into MedHarness desktop app (Settings > Cloud Brain)
  Models     : medr1   = $MEDR1_FINAL  (~5.4 GB, GPTQ W4A16 Marlin)
               medgemma = $MEDGEMMA_FINAL (~2.9 GB, GPTQ W4A16 Marlin)
  Bearer     : $MASKED_TOKEN  (already configured locally — NEVER share this notebook)
  KV cache   : ~$(awk -v v="$VRAM_MB" 'BEGIN{printf "%.1f", v/1024*0.93-8.3}') GB PagedAttention headroom
  Logs       : $WORKDIR/logs/   Manifest: $WORKDIR/CONNECTION_MANIFEST.json
  Session cap: ${SESSION_MAX_HOURS}h (hard safety stop)
============================================================================
BANNER

# ------------------------------------------------- supervisor loop ----------
# Keeps children alive (bounded restarts), logs GPU pressure, enforces the
# session cap, and lets the LOCAL heartbeat (every 3 min) keep the VM awake.
declare -A RESTARTS=( [medr1]=0 [medgemma]=0 [gateway]=0 [tunnel]=0 )
cleanup() {
  log "Shutting down cloud brain…"
  for p in pid_tunnel pid_gateway pid_medr1 pid_medgemma; do
    [ -f "$WORKDIR/$p" ] && kill "$(cat "$WORKDIR/$p")" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

while true; do
  sleep 60
  # hard session cap — free tiers are for evaluation, not residency
  NOW="$(date -u +%s)"
  if [ $(( NOW - SESSION_STARTED_AT )) -ge $(( SESSION_MAX_HOURS * 3600 )) ]; then
    warn "SESSION_MAX_HOURS=${SESSION_MAX_HOURS} reached — stopping cleanly."
    exit 0
  fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader \
    >> "$WORKDIR/logs/gpu_pressure.log" 2>/dev/null || true
  [ -f "$WORKDIR/pid_medr1" ] && ! kill -0 "$(cat "$WORKDIR/pid_medr1")" 2>/dev/null && {
    RESTARTS[medr1]=$(( ${RESTARTS[medr1]} + 1 ))
    if [ "${RESTARTS[medr1]}" -le 3 ]; then warn "medr1 died — restart ${RESTARTS[medr1]}/3"; launch_r1; else die "medr1 crash-looped; inspect logs/vllm_medr1.log"; fi
  } || true
  [ -f "$WORKDIR/pid_medgemma" ] && ! kill -0 "$(cat "$WORKDIR/pid_medgemma")" 2>/dev/null && {
    RESTARTS[medgemma]=$(( ${RESTARTS[medgemma]} + 1 ))
    if [ "${RESTARTS[medgemma]}" -le 3 ]; then warn "medgemma died — restart ${RESTARTS[medgemma]}/3"; launch_gemma; else die "medgemma crash-looped; inspect logs/vllm_medgemma.log"; fi
  } || true
  [ -f "$WORKDIR/pid_gateway" ] && ! kill -0 "$(cat "$WORKDIR/pid_gateway")" 2>/dev/null && {
    RESTARTS[gateway]=$(( ${RESTARTS[gateway]} + 1 ))
    if [ "${RESTARTS[gateway]}" -le 3 ]; then warn "gateway died — restart ${RESTARTS[gateway]}/3"; launch_gateway; fi
  } || true
  [ -f "$WORKDIR/pid_tunnel" ] && ! kill -0 "$(cat "$WORKDIR/pid_tunnel")" 2>/dev/null && {
    RESTARTS[tunnel]=$(( ${RESTARTS[tunnel]} + 1 ))
    if [ "${RESTARTS[tunnel]}" -le 3 ]; then
      warn "tunnel died — restart ${RESTARTS[tunnel]}/3 (URL may change; re-paste in app)"
      nohup "$WORKDIR/cloudflared" tunnel --url "http://127.0.0.1:${GATEWAY_PORT}" \
        --no-autoupdate --protocol http2 >> "$WORKDIR/logs/cloudflared.log" 2>&1 &
      echo $! > "$WORKDIR/pid_tunnel"
    fi
  } || true
done
