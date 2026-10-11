# MedHarness

**Zero-cost clinical AI workstation.** A lightweight desktop harness + a free
Google Colab T4 "cloud brain" running **Med-R1-8B** (clinical reasoning) and
**MedGemma-4B** (imaging/multimodal) simultaneously in VRAM — quantized
**GPTQ W4A16 Marlin** — behind a bearer-authenticated Cloudflare tunnel.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KeshavCracks/openagent/blob/main/colab/MedHarness_CloudBrain.ipynb)
[![Download](https://img.shields.io/badge/download-Releases-0E7C66)](https://github.com/KeshavCracks/openagent/releases/latest)

## Why MedHarness

- **Zero compute cost, zero setup** — no Docker, no GPU ownership, no terminal skills.
- **Both models hot in VRAM** (~8.3 GB weights on a 16 GB T4) — no loading stalls mid-session.
- **Deterministic clinical math** — CrCl, eGFR (CKD-EPI 2021), BSA, pediatric mg/kg, IV drip rates, 4-2-1 fluids are computed locally by a verified engine; the LLM never does arithmetic.
- **Fast on free hardware** — client-side context packing (dedup + boilerplate pruning) cuts tunnel time-to-first-token.
- **Safety gates** — anti-rationalization quality gates (`/spec /plan /build /test /review /constraints`), progressive-disclosure checklists, diagnosis-without-differential blocking.
- **Private by design** — CSPRNG bearer token closes the public tunnel, remote request logging disabled, Groq is used only as a headless DevOps agent (never for clinical traffic).

## Quickstart (3 steps)

1. **Get the harness** — download `medical-harness-*.zip` from
   [**Releases**](https://github.com/KeshavCracks/openagent/releases/latest),
   unzip, then: `pip install -r requirements.txt && python3 -m app.main`
   (opens `http://127.0.0.1:8787`).
2. **Start the cloud brain** — open the
   [Colab notebook](https://colab.research.google.com/github/KeshavCracks/openagent/blob/main/colab/MedHarness_CloudBrain.ipynb)
   on a **T4 runtime** and run both cells (~3–8 min). It prints a session
   token and a `https://….trycloudflare.com` URL.
3. **Connect** — paste the token and tunnel URL into the app
   (Settings → Cloud Brain → Connect). The status chip turns **healthy**
   and the 3-minute heartbeat keeps the VM awake. Chat.

## Repository layout

| Path | Purpose |
|---|---|
| `app/` | Desktop harness: FastAPI server, settings/token management, 4px-grid WCAG AA chat UI |
| `core/` | `cloud_bootstrap.sh` (remote deploy) · Groq orchestrator · HTTP/2 tunnel client · heartbeat · deterministic math engine |
| `plugins/` | Quality gates · context optimizer · editorial SVG diagrams (offline, zero Mermaid) · PubMed/WHO/CDC grounding |
| `colab/` | One-click cloud-brain notebook |
| `docs/ARCHITECTURE.md` | Full architecture, constraint enforcement points, quantization recipe |
| `tests/` | 36 offline tests (math accuracy, token reduction, tunnel auth) |

## Security notes

- Session token: CSPRNG, persisted `0600`; validated by the gateway **and** both vLLM engines.
- Cloudflare interstitial bypass via `curl` user-agent on every request.
- Public grounding module is opt-in, read-only, host-allowlisted, and never carries session data.

> **Decision support for trained clinicians — not a medical device.** Verify all outputs against institutional protocols.
