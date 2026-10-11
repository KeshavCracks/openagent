# /build — Gated execution (invoked guideline)

Execute ONLY the current /plan, step by step:

1. Restate the current step and its verification check before doing it.
2. Perform the step.
3. Run the verification. If it fails: stop, report, propose a fix — do NOT
   silently continue (the gate will demand user confirmation).
4. Emit a one-line checkpoint log: `✓ Step N verified by <check>`.

Forbidden under this gate:
- Modifying scope without a new /spec.
- Substituting "estimated" numbers for computed ones (math engine is the
  only arithmetic authority).
- Marking steps complete without their verification artifact.
