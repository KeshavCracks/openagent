"""Aggressive context packing — cut TTFT before bytes cross the tunnel.

Clinical histories, EHR dumps, and lab exports are saturated with
administrative boilerplate (confidentiality notices, page furniture,
repeated signature blocks). Every token pruned here is a token the remote
T4 never has to prefill, so packing happens CLIENT-SIDE before upload.

Pipeline: normalize → strip boilerplate → deduplicate repeated lines →
collapse whitespace → chunk with metadata headers → report stats.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Pattern

# ----------------------------------------------------------- boilerplate ---
_BOILERPLATE: tuple = (
    r"(?i)^confidentiality notice\b.*",
    r"(?i)^this (e-?mail|message|communication|fax)( .*)?\bcontains\b.*",
    r"(?i)^if you (are not|have received|believe).*please (notify|delete|ignore).*",
    r"(?i)^this (e-?mail|message) is (intended|for the (exclusive|use)).*",
    r"(?i)^privileged and confidential.*",
    r"(?i)^disclaimer:?(\s|$).*",
    r"(?i)^notice:?(\s|$).*this.*(phi|protected health).*",
    r"(?i)^page \d+( of \d+)?$",
    r"^\s*[-=_*~]{4,}\s*$",
    r"(?i)^printed (on|by|from).*",
    r"(?i)^\s*(sent|received) from my .*",
)
_BOILERPLATE_RE: tuple = tuple(re.compile(p) for p in _BOILERPLATE)


@dataclass
class PackedContext:
    text: str
    original_chars: int
    packed_chars: int
    chunks: List[str] = field(default_factory=list)
    removed_lines: int = 0

    # ~4 chars per token; good enough for budget decisions, never for billing.
    @property
    def original_tokens(self) -> int:
        return self.original_chars // 4

    @property
    def packed_tokens(self) -> int:
        return self.packed_chars // 4

    @property
    def saved_pct(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return round(100.0 * (1.0 - self.packed_chars / self.original_chars), 1)

    def stats(self) -> dict:
        return {
            "original_tokens": self.original_tokens,
            "packed_tokens": self.packed_tokens,
            "saved_pct": self.saved_pct,
            "chunks": len(self.chunks),
            "removed_lines": self.removed_lines,
        }


# --------------------------------------------------------------- helpers ---
_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")


def strip_boilerplate(text: str) -> tuple[str, int]:
    removed = 0
    out_lines: List[str] = []
    for line in text.splitlines():
        if any(rx.match(line.strip()) for rx in _BOILERPLATE_RE):
            removed += 1
            continue
        out_lines.append(line)
    return "\n".join(out_lines), removed


def dedupe_lines(text: str) -> tuple[str, int]:
    """Drop exact repeated lines beyond first occurrence (lab tables, sig blocks)."""
    seen: set = set()
    out: List[str] = []
    removed = 0
    for line in text.splitlines():
        key = line.strip().lower()
        if key and key in seen:
            removed += 1
            continue
        if key:
            seen.add(key)
        out.append(line)
    return "\n".join(out), removed


def chunk(text: str, chunk_chars: int = 1600) -> List[str]:
    """Structure-preserving chunking on paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for p in paragraphs:
        plen = len(p)
        if buf and size + plen > chunk_chars:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        # single pathological paragraph longer than the budget: hard-split
        while plen > chunk_chars:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, size = [], 0
            chunks.append(p[:chunk_chars])
            p = p[chunk_chars:]
            plen = len(p)
        buf.append(p)
        size += plen
    if buf:
        chunks.append("\n\n".join(buf))
    return [f"[CHUNK {i + 1}/{len(chunks)}]\n{c}" for i, c in enumerate(chunks)]


# ------------------------------------------------------------- entrypoint --
def pack(text: str, chunk_chars: int = 1600) -> PackedContext:
    """Full packing pipeline. Never reorders clinical content; only removes."""
    original_chars = len(text)
    if not text.strip():
        return PackedContext(text="", original_chars=original_chars, packed_chars=0)
    stripped, removed_bp = strip_boilerplate(text)
    deduped, removed_dup = dedupe_lines(stripped)
    collapsed = _BLANK_RE.sub("\n\n", _WS_RE.sub(" ", deduped)).strip()
    # single short notes stay unlabeled — the packer must never expand text
    if len(collapsed) <= chunk_chars:
        return PackedContext(
            text=collapsed,
            original_chars=original_chars,
            packed_chars=len(collapsed),
            chunks=[collapsed] if collapsed else [],
            removed_lines=removed_bp + removed_dup,
        )
    chunks = chunk(collapsed, chunk_chars)
    packed = "\n\n".join(chunks)
    return PackedContext(
        text=packed,
        original_chars=original_chars,
        packed_chars=len(packed),
        chunks=chunks,
        removed_lines=removed_bp + removed_dup,
    )


def pack_message(content: str, chunk_chars: int = 1600) -> tuple[str, dict]:
    """Convenience wrapper returning (packed_text, stats) for chat history."""
    ctx = pack(content, chunk_chars)
    return ctx.text, ctx.stats()
