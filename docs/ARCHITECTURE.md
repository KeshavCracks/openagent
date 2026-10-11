# MedHarness — Zero-Cost Clinical AI Desktop Harness

Production-grade, zero-setup, zero-compute-cost medical AI workstation.
**Split-brain architecture**: a lightweight local client (this repo) + a free
Google Colab/Kaggle NVIDIA T4 "cloud brain" running both clinical models hot
in VRAM behind an authenticated Cloudflare quick tunnel.

> Decision support for trained clinicians. Not a medical device. Arithmetic
> is computed by the deterministic local engine — never by the model.

---

## Architecture

```
┌────────────── LOCAL HARNESS (Linux desktop) ──────────────┐
│  chat UI (4px grid, WCAG AA)                              │
│  math interceptor → deterministic engine (GFR/BSA/drips)  │
│  context packer (dedup + boilerplate prune → low TTFT)    │
│  quality gates (/spec /plan /build /test /review /…)      │
│  heartbeat (3-min keep-alive)  bypass headers (curl UA)   │
└───────────────────────┬───────────────────────────────────┘
                        │ HTTP/2 + Bearer SESSION_BEARER_TOKEN
                        ▼  (trycloudflare quick tunnel)
┌────────────── REMOTE T4 CLOUD BRAIN ──────────────────────┐
│  gateway :8000 (bearer auth → alias routing)              │
│  vLLM :8001 Med-R1-8B   GPTQ W4A16 Marlin  ~5.4 GB        │
│  vLLM :8002 MedGemma-4B GPTQ W4A16 Marlin  ~2.9 GB        │
│  --enable-prefix-caching · --disable-log-requests (PHI)   │
└───────────────────────────────────────────────────────────┘
        ▲
        │ headless DevOps ONLY (parameter JSON — never clinical data)
   Groq API
```

**Locked constraints (enforced mechanically, not by convention):**

| Constraint | Enforcement point |
|---|---|
| GPTQ W4A16 Marlin ONLY — AWQ/FP8 forbidden | `cloud_bootstrap.sh` preflight aborts on non-GPTQ `quantization_config`; `orchestrator` audits Groq proposals and rejects AWQ/FP8 |
| Both models hot in VRAM (~8.3 GB weights + KV cache) | T4-tuned `--gpu-memory-utilization` split, zero swap |
| LLM never computes clinical arithmetic | `core/math_engine.py` intercepts math intent; verified block injected as ground truth |
| Groq never sees clinical traffic | Groq only emits deployment *parameters* (JSON); execution is the audited canonical script |
| Public tunnel closed to strangers | Local CSPRNG `SESSION_BEARER_TOKEN` (0600 file) validated by gateway AND vLLM `--api-key` |
| Cloudflare interstitial breaks JSON | every request carries `User-Agent: curl/8.0.0`; one browser-UA retry ladder |
| Colab idle-reaping | `core/heartbeat.py` pings `/health` every 3 min ± jitter |
| No PHI in notebook logs | `--disable-log-requests` on both engines |

## Quickstart (end user — no terminal knowledge needed)

```bash
pip install -r requirements.txt   # five packages, no Docker
./run.sh                          # opens http://127.0.0.1:8787
```

1. **Settings → Cloud Brain → Deploy cloud brain.** (Optional: paste a free
   Groq key first — it is used ONLY as a headless DevOps engineer.)
2. Copy the single generated cell into a **T4** Google Colab session and run
   it. Wait for the `CLOUD BRAIN — ONLINE` banner (3–8 min).
3. Paste the printed `https://….trycloudflare.com` URL into Settings →
   Connect. Status chip turns **healthy** (heartbeat starts automatically).
4. Chat. Use `/spec /plan /build /test /review /constraints` workflows;
   clinical math is intercepted and verified locally before the model sees it.

## Obtaining GPTQ W4A16 checkpoints (one-time, if no public mirror exists)

The bootstrap preflight **refuses** non-GPTQ artifacts (exit code 3) — this is
the Quantization-Trap guard (AWQ ≈ −16% on CareQA for Gemma3/MedGemma; the T4
has no FP8 tensor cores). Quantize once on CPU and point the cell at it:

```python
from gptqmodel import GPTQModel, QuantizeConfig   # or auto-gptq
qc = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False)  # W4A16
model = GPTQModel.load("google/medgemma-4b-it", qc)
model.quantize(calibration_texts)                  # ~512 clinical-ish samples
model.save("./medgemma-4b-gptq")                   # → upload / mount
```

Then set `MEDGEMMA_GPTQ_REPO=/path/to/medgemma-4b-gptq` (and the Med-R1
equivalent) in the bootstrap env — the deployment pack accepts any HF repo id
or local directory path.

## Project layout

```text
medical-harness/
├── app/            # entrypoint, config/token mgmt, 4px-grid UI
├── core/           # orchestrator · cloud_bootstrap.sh · tunnel_client · heartbeat · math_engine
├── plugins/        # quality_gates · context_optimizer · editorial_diagrams · agent_reach
└── tests/          # math interceptor (0% arithmetic hallucination) · context packing · tunnel auth
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q     # 36 tests, fully offline (MockTransport)
```

## Packaging notes (PyInstaller / AppImage)

All local modules stick to `pathlib.Path` + `os.environ` + `asyncio` — no
platform-specific shell quirks, no `/tmp` hardcoding, no Docker. The only
subprocess is `run.sh`, which simply invokes `python3 -m app.main`. A frozen
build needs: `pyinstaller --collect-all app --collect-all core --collect-all
plugins app/main.py` plus the `app/ui` directory as a data file.

## Privacy & security model

* Session token: CSPRNG (`secrets.token_urlsafe(32)`), persisted 0600 under
  `~/.local/state/medical-harness/`. Never printed in full on the remote side
  (banner shows first 8 chars).
* Gateway rejects unauthenticated requests with 401 before any engine touch;
  both engines double-check via vLLM `--api-key`.
* Outbound module (`agent_reach`) is opt-in, GET-only, host-allowlisted
  (PubMed E-utilities, WHO, CDC), and carries only query terms — the quality
  gate hard-denies any payload containing PHI markers.
* Remote request logging disabled; context packing happens client-side, so
  the uploaded prompt is already minimal.
