"""Slash commands, progressive disclosure, and anti-rationalization gates.

Design (mapped from addyosmani/agent-skills + vercel-labs/agent-skills):

* Progressive disclosure — only the DEFINITIONS table (name, summary,
  when-to-use — a few hundred bytes each) is loaded into RAM at thread
  start. The fully expanded clinical checklists (guidelines/*.md) are read
  from disk ONLY when a command is actually invoked, preserving local RAM
  and context boundaries.

* Anti-rationalization gates — validation assertions that pause execution
  and demand explicit user confirmation when the agent tries to skip
  verification steps, bypass differential-diagnosis structure, or ignore
  clinical edge cases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_GUIDELINE_DIR = Path(__file__).with_name("guidelines")

# ---------------------------------------------------------------------------
# Lightweight definitions — loaded at kernel boot (progressive disclosure L0)
# ---------------------------------------------------------------------------
DEFINITIONS: Dict[str, Dict[str, str]] = {
    "/spec": {
        "summary": "Draft a bounded clinical/workflow specification before any build step.",
        "when_to_use": "Before implementing a new pathway, integration, or document template.",
    },
    "/plan": {
        "summary": "Produce a stepwise plan with verification checkpoints.",
        "when_to_use": "Multi-step clinical or engineering tasks where order matters.",
    },
    "/build": {
        "summary": "Execute the plan with quality gates enforced at each checkpoint.",
        "when_to_use": "Only after a /spec and /plan exist (gate-enforced).",
    },
    "/test": {
        "summary": "Run verification pass and enumerate untested assumptions.",
        "when_to_use": "Before any output is treated as final; never skippable (gate-enforced).",
    },
    "/review": {
        "summary": "Structured self-review: differentials, red flags, edge cases.",
        "when_to_use": "Before signing off on diagnostic or dosing content.",
    },
    "/constraints": {
        "summary": "Show the locked architecture + clinical safety constraints (answered locally).",
        "when_to_use": "Any time the agent or user is tempted to bypass a safeguard.",
    },
}


def load_guideline(command: str) -> str:
    """Progressive disclosure L1: read the full checklist ONLY on invocation."""
    name = command.lstrip("/")
    path = _GUIDELINE_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"No guideline file for {command}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
@dataclass
class GateBlock(Exception):
    """Raised/paused when an anti-rationalization rule trips."""

    gate: str
    reason: str
    requires_user_confirmation: bool = True
    details: List[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.gate}] {self.reason}"


_DIAGNOSIS_TRIGGERS = re.compile(
    r"\b(therefore the diagnosis is|most likely diagnosis|final diagnosis|"
    r"this confirms that the patient has)\b", re.I)
_DIFFERENTIAL_MARKERS = re.compile(r"\bdifferential\b|\bddx\b|alternative diagnoses|rule ?out", re.I)
_REDFLAG_MARKERS = re.compile(r"\bred flags?\b|\bemergen|referral|escalat", re.I)
_PHI_MARKERS = re.compile(r"\b(MRN|date of birth|DOB|SSN)\b", re.I)


def validate_pre_action(action: str, context: Optional[Dict[str, str]] = None) -> None:
    """Hard assertion layer before an action executes. Raises GateBlock."""
    context = context or {}
    if action == "skip_verification":
        raise GateBlock(
            gate="verification",
            reason="Skipping verification is not allowed. Explicit user confirmation required.",
        )
    if action == "build_without_spec":
        raise GateBlock(
            gate="build",
            reason="Refusing to /build without a /spec and /plan. Run them first.",
        )
    if action == "deploy_remote":
        if not context.get("token"):
            raise GateBlock(
                gate="deploy",
                reason="Remote deployment attempted without a session token.",
                requires_user_confirmation=False,
            )
    if action == "external_network":
        if _PHI_MARKERS.search(context.get("payload", "")):
            raise GateBlock(
                gate="privacy",
                reason="Outbound payload contains PHI markers; public grounding module must never receive session data.",
                requires_user_confirmation=False,  # hard deny, never confirmable
                details=["Strip identifiers or disable the module."],
            )


def validate_output(answer: str) -> Optional[GateBlock]:
    """Post-generation audit: structural clinical-safety heuristics (v1).

    Returns a GateBlock when the model asserts a diagnosis without a
    differential or red-flag section — the execution engine pauses and asks
    the user to confirm before the answer is presented as final.
    """
    if _DIAGNOSIS_TRIGGERS.search(answer):
        missing: List[str] = []
        if not _DIFFERENTIAL_MARKERS.search(answer):
            missing.append("differential diagnosis list (≥3 alternatives)")
        if not _REDFLAG_MARKERS.search(answer):
            missing.append("red-flag / escalation guidance")
        if missing:
            return GateBlock(
                gate="diagnostic-structure",
                reason="Model asserted a diagnosis without required structure.",
                details=[f"Missing: {', '.join(missing)}.", "Confirm to accept, or regenerate with structure."],
            )
    return None


# ---------------------------------------------------------------------------
# Slash command handling (kernel entrypoint)
# ---------------------------------------------------------------------------
def handle_slash(message: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse a leading slash command.

    Returns (command, guideline_text, local_reply):
      * command is None if the message is not a slash command.
      * /constraints is answered LOCALLY (no remote call, no PHI leaves).
      * other commands return guideline_text to augment the system prompt
        (read from disk at this exact moment — progressive disclosure).
    """
    stripped = message.strip()
    if not stripped.startswith("/"):
        return None, None, None
    parts = stripped.split(maxsplit=1)
    command = parts[0].lower()
    if command not in DEFINITIONS:
        return command, None, (
            f"Unknown command `{command}`. Available: {', '.join(sorted(DEFINITIONS))}"
        )
    if command == "/constraints":
        from app.config import QUANT_LOCK  # local import: config owns constraints

        local_reply = (
            "## Locked constraints\n"
            f"* **Quantization:** {QUANT_LOCK['quantization']} — {QUANT_LOCK['reason']}\n"
            "* **Clinical arithmetic:** never generated by the LLM; intercepted and computed by the "
            "deterministic math engine, injected as ground truth.\n"
            "* **Groq scope:** headless DevOps provisioning only; zero clinical traffic to Groq.\n"
            "* **Tunnel:** Cloudflare quick tunnel, bearer-authenticated, curl-UA bypass headers, "
            "3-minute keep-alive heartbeat.\n"
            "* **Diagrams:** self-contained 4px-grid HTML+SVG; zero Mermaid; offline; WCAG AA.\n"
            "* **Context:** client-side deduplication and boilerplate pruning before upload (TTFT).\n"
            "* **Guidelines:** progressive disclosure — checklists load only when invoked."
        )
        return command, None, local_reply
    return command, load_guideline(command), None
