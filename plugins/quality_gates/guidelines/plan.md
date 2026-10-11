# /plan — Stepwise plan with checkpoints (invoked guideline)

Format each step as: `Step N — <verb phrase>` followed by
- **Produces:** concrete artifact
- **Verified by:** specific check (test, diff, calculation, second look)
- **Rollback:** what to do if verification fails

Hard rules:
- Every step that touches dosing, calculation, or diagnosis MUST end with a
  verification checkpoint. Skipping one requires explicit user confirmation
  (the anti-rationalization gate will pause otherwise).
- Order steps so that the cheapest destructive-failure detection happens first.
- Cap the plan at 9 steps; beyond that, split into sub-plans.
