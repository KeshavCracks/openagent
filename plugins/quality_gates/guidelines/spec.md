# /spec — Bounded specification (invoked guideline)

Produce a specification with EXACTLY these sections:

1. **Problem** — one paragraph, no solutions.
2. **Actors & data** — who touches what; classify each datum as PHI or non-PHI.
3. **Scope boundaries** — explicit "out of scope" list (≥3 items).
4. **Interfaces** — inputs, outputs, failure modes per interface.
5. **Verification plan** — how each requirement will be checked (map 1:1).
6. **Clinical safety notes** — edge cases, contraindications, escalation paths.

Rules:
- No implementation details in this phase.
- Every requirement MUST be testable; untestable ones get split until testable.
- If any datum is PHI, name the module boundary that keeps it local.
