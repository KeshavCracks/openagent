# /constraints — Locked constraints (invoked guideline, answered locally)

This command never contacts the remote models; the kernel answers it
locally from `app.config`. Constraints shown:

- GPTQ W4A16 Marlin quantization (AWQ/FP8 forbidden — T4 has no FP8 cores;
  AWQ costs up to 16% on Gemma3/MedGemma medical benchmarks).
- Dual-model VRAM residency: ~5.4 GB (Med-R1-8B) + ~2.9 GB (MedGemma-4B),
  remaining budget to PagedAttention KV cache, zero model swapping.
- Deterministic arithmetic interceptor: the LLM never computes medical math.
- Groq = headless DevOps only; clinical payloads never leave the tunnel path.
- Cloudflare interstitial bypass + bearer token on every request.
- 3-minute keep-alive heartbeat against Colab/Kaggle idle reaping.
- Prefix caching + SSE streaming with collapsible `<think>` accordion.
- Zero Mermaid; self-contained 4px-grid SVG; offline; WCAG AA.
