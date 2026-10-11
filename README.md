# MedHarness

**Zero-cost clinical AI workstation.** A lightweight desktop harness + a free
Google Colab T4 "cloud brain" running **Med-R1-8B** (clinical reasoning) and
**MedGemma-4B** (imaging/multimodal) simultaneously in VRAM — quantized
**GPTQ W4A16 Marlin** — behind a bearer-authenticated Cloudflare tunnel.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KeshavCracks/openagent/blob/main/colab/MedHarness_CloudBrain.ipynb)
[![Download](https://img.shields.io/badge/download-Releases-0E7C66)](https://github.com/KeshavCracks/openagent/releases/latest)

## Why MedHarness

- **Zero compute cost, zero setup** — no Docker, no GPU ownership, no terminal skills (double-click `run.bat` / `run.sh`).
- **Both models hot in VRAM** (~8.3 GB weights on a free Colab T4) — no loading stalls mid-session, with live model-health chips in the UI.
- **14 deterministic clinical calculators** — CrCl, eGFR (CKD-EPI 2021), BSA, pediatric mg/kg, IV drip/mcg-kg-min rates, 4-2-1 fluids, anion gap, corrected Na, MAP, albumin-corrected calcium, CHA2DS2-VASc, Wells (DVT & PE), CURB-65 — computed locally by a verified engine; the LLM never does arithmetic.
- **Fast on free hardware** — client-side context packing (dedup + boilerplate pruning) cuts tunnel time-to-first-token.
- **Safety gates** — anti-rationalization quality gates (`/spec /plan /build /test /review /constraints`), progressive-disclosure checklists, diagnosis-without-differential blocking.
- **Consult-ready output** — markdown tables and headings in answers, one-click printable PDF summary, dark mode for on-call.
- **Evidence at hand** — opt-in, read-only PubMed lookups you can drop into the conversation as citations.
- **Private by design** — CSPRNG bearer token closes the public tunnel, remote request logging disabled, Groq is used only as a headless DevOps agent (never for clinical traffic).

## Quickstart (3 steps)

1. **Get the harness** — download `medical-harness-*.zip` from
   [**Releases**](https://github.com/KeshavCracks/openagent/releases/latest),
   unzip, then double-click `run.bat` (Windows) or `bash run.sh` (Linux/macOS)
   — or manually: `pip install -r requirements.txt && python3 -m app.main`
   (opens `http://127.0.0.1:8787`).
2. **Start the cloud brain** — open the
   [Colab notebook](https://colab.research.google.com/github/KeshavCracks/openagent/blob/main/colab/MedHarness_CloudBrain.ipynb)
   on a **T4 runtime** and run both cells (~3–8 min). It prints a session
   token and a `https://….trycloudflare.com` URL.
3. **Connect** — paste the token and tunnel URL into the app
   (Settings → Cloud Brain → Connect). The status chip turns **healthy**
   and the 3-minute heartbeat keeps the VM awake. Chat.

New here? Read [SETUP.md](SETUP.md) — a plain-language walkthrough with an FAQ
and troubleshooting table.

## Repository layout

| Path | Purpose |
|---|---|
| `app/` | Desktop harness: FastAPI server, settings/token management, 4px-grid WCAG AA chat UI |
| `core/` | `cloud_bootstrap.sh` (remote deploy) · Groq orchestrator · HTTP/2 tunnel client · heartbeat · deterministic math engine |
| `plugins/` | Quality gates · context optimizer · editorial SVG diagrams (offline, zero Mermaid) · PubMed/WHO/CDC grounding |
| `colab/` | One-click cloud-brain notebook |
| `docs/ARCHITECTURE.md` | Full architecture, constraint enforcement points, quantization recipe |
| `tests/` | 57 offline tests (math accuracy incl. score scales, token reduction, tunnel auth, model panel & evidence endpoints) |

## Security notes

- Session token: CSPRNG, persisted `0600`; validated by the gateway **and** both vLLM engines.
- Cloudflare interstitial bypass via `curl` user-agent on every request.
- Public grounding module is opt-in, read-only, host-allowlisted, and never carries session data.

> **Decision support for trained clinicians — not a medical device.** Verify all outputs against institutional protocols.
