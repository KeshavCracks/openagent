# /test — Verification pass (invoked guideline, never skippable)

Run through ALL applicable layers and report PASS/FAIL per item:

1. **Deterministic math** — every numeric claim traces to the math engine
   output block (no model-computed arithmetic anywhere).
2. **Differential structure** — diagnostic content lists ≥3 alternatives
   with discriminating features and red flags.
3. **Context integrity** — packed context preserved the decisive facts
   (spot-check labs, meds, allergies survived pruning).
4. **Privacy** — nothing PHI left the machine (tunnel payload = packed
   clinical context only to OUR gateway; Groq received zero clinical data).
5. **Untested assumptions** — enumerate anything you could not verify.

If any FAIL: the answer is not final. State what must change.
