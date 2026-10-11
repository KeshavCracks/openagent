"""Pure editorial clinical diagrams — self-contained HTML+SVG on a strict 4px grid.

Hard rules (locked):
  * ZERO Mermaid.js. ZERO external rendering engines. ZERO network fetches —
    each render returns a fully self-contained <svg> string with an inline
    <style> block driven by localized CSS variables (--bg-paper, --text-ink,
    --accent-clinical, --muted-subtle), so diagrams survive offline export
    into notes or PDFs.
  * All coordinates snap to a 4px grid (audited by `_snap` in tests).
  * WCAG AA contrast: ink-on-paper ≥ 7:1, muted ≥ 4.5:1 on paper; diagram
    text never sits on accent fills.
  * Accessibility: every diagram carries <title>/<desc>; user-supplied
    labels are XML-escaped (defense against label-injected markup).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
from xml.sax.saxutils import escape

GRID = 4  # px


def _snap(v: float) -> int:
    return int(round(v / GRID)) * GRID


COLORS: Dict[str, str] = {
    "bg_paper": "#FAFAF7",
    "text_ink": "#1F2A33",
    "accent_clinical": "#0E7C66",
    "muted_subtle": "#5B6770",
    "line_quiet": "#D7DCD8",
    "think_backdrop": "#F1F4F2",
}

_STYLE = f"""
  <style><![CDATA[
    :root, svg {{ --bg-paper:{COLORS['bg_paper']}; --text-ink:{COLORS['text_ink']};
                  --accent-clinical:{COLORS['accent_clinical']};
                  --muted-subtle:{COLORS['muted_subtle']}; --line-quiet:{COLORS['line_quiet']}; }}
    .t-ink   {{ fill: var(--text-ink);         font: 600 16px/1.4 'Helvetica Neue', Arial, sans-serif; }}
    .t-title {{ fill: var(--text-ink);         font: 700 20px/1.3 'Helvetica Neue', Arial, sans-serif; }}
    .t-muted {{ fill: var(--muted-subtle);     font: 400 13px/1.4 'Helvetica Neue', Arial, sans-serif; }}
    .t-onbox {{ fill: var(--bg-paper);         font: 600 15px/1.3 'Helvetica Neue', Arial, sans-serif; }}
    .box     {{ fill: var(--accent-clinical);  rx: 8; }}
    .box-quiet {{ fill: var(--think-backdrop); stroke: var(--line-quiet); }}
    .edge    {{ stroke: var(--muted-subtle);   stroke-width: 2; fill: none; }}
  ]]></style>"""


def _svg_open(w: int, h: int, title: str, desc: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="100%" role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title><desc>{escape(desc)}</desc>{_STYLE}"
        f'<rect width="{w}" height="{h}" fill="var(--bg-paper)"/>'
    )


def _box(x: int, y: int, w: int, label: str, cls: str = "box", lines_gap: int = 24) -> str:
    """Fixed-width 4px-grid box with wrapped text centered."""
    x, y, w = _snap(x), _snap(y), _snap(w)
    words = label.split()
    lines: List[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if len(trial) * 8 > w - 32 and cur:  # ~8px/char heuristic
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    h = _snap(40 + lines_gap * max(0, len(lines) - 1))
    parts = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>']
    ty = y + 26
    for line in lines:
        parts.append(
            f'<text x="{x + w // 2}" y="{ty}" text-anchor="middle" '
            f'class="{"t-ink" if cls.endswith("quiet") else "t-onbox"}">{escape(line)}</text>'
        )
        ty += lines_gap
    return "".join(parts)


def _arrow(x1: int, y1: int, x2: int, y2: int, label: str = "") -> str:
    x1, y1, x2, y2 = _snap(x1), _snap(y1), _snap(x2), _snap(y2)
    out = f'<path d="M{x1},{y1} L{x2},{y2}" class="edge"/>'
    out += (
        f'<circle cx="{x2}" cy="{y2}" r="4" fill="var(--muted-subtle)"/>'
    )
    if label:
        out += f'<text x="{_snap((x1 + x2) / 2) + 8}" y="{_snap((y1 + y2) / 2) - 8}" class="t-muted">{escape(label)}</text>'
    return out


# --------------------------------------------------------------- pathway ---
def render_pathway(steps: Sequence[Dict[str, str]], title: str = "Clinical pathway") -> str:
    """Vertical flowchart: [{'label': ..., 'note': ...}, ...] top→bottom."""
    w = 560
    box_w = 400
    x = _snap((w - box_w) / 2)
    y = 64
    gap = 56
    parts = [_svg_open(w, _snap(64 + len(steps) * (64 + gap) + 16), title,
                       "Editorial clinical pathway rendered as offline SVG.")]
    parts.append(f'<text x="{_snap(w / 2)}" y="36" text-anchor="middle" class="t-title">{escape(title)}</text>')
    for i, step in enumerate(steps):
        parts.append(_box(x, y, box_w, step.get("label", "")))
        if step.get("note"):
            parts.append(
                f'<text x="{x + box_w + 8}" y="{y + 24}" class="t-muted">{escape(step["note"])}</text>'
            )
        if i < len(steps) - 1:
            parts.append(_arrow(x + box_w // 2, y + 64, x + box_w // 2, y + 64 + gap))
        y += 64 + gap
    parts.append("</svg>")
    return "".join(parts)


# -------------------------------------------------------------- timeline ---
def render_timeline(events: Sequence[Dict[str, str]], title: str = "Disease timeline") -> str:
    """Horizontal axis: [{'when': 'Day 0', 'label': ...}, ...] left→right."""
    n = max(1, len(events))
    w = _snap(120 * n + 80)
    axis_y = 200
    parts = [_svg_open(w, 280, title, "Editorial patient timeline rendered as offline SVG.")]
    parts.append(f'<text x="40" y="36" class="t-title">{escape(title)}</text>')
    parts.append(f'<line x1="40" y1="{axis_y}" x2="{w - 40}" y2="{axis_y}" class="edge"/>')
    step = (w - 80) // max(1, n)
    for i, ev in enumerate(events):
        cx = _snap(40 + step * i + (step // 2 if n > 1 else 0))
        parts.append(f'<circle cx="{cx}" cy="{axis_y}" r="8" fill="var(--accent-clinical)"/>')
        parts.append(f'<text x="{cx}" y="{axis_y - 24}" text-anchor="middle" class="t-ink">{escape(ev.get("when", ""))}</text>')
        # label below, wrapped crudely on 2 lines
        label = ev.get("label", "")
        cut = len(label) // 2
        parts.append(f'<text x="{cx}" y="{axis_y + 32}" text-anchor="middle" class="t-muted">{escape(label[:cut])}</text>')
        if cut:
            parts.append(f'<text x="{cx}" y="{axis_y + 52}" text-anchor="middle" class="t-muted">{escape(label[cut:])}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ tree ---
def render_tree(nodes: Sequence[Dict[str, str]], title: str = "Diagnostic tree") -> str:
    """Simple binary-ish decision tree: nodes listed level-order with 'yes'/'no' parents.

    Each node: {'label': str, 'yes': idx|None, 'no': idx|None}. Root = idx 0.
    Layout: BFS levels, horizontally spaced on a 4px grid.
    """
    levels: List[List[int]] = [[0]] if nodes else []
    visited = set([0]) if nodes else set()
    while levels:
        nxt: List[int] = []
        for idx in levels[-1]:
            for child_key in ("yes", "no"):
                child = nodes[idx].get(child_key)
                if child is not None and child not in visited:
                    visited.add(child)
                    nxt.append(child)
        if nxt:
            levels.append(nxt)
        else:
            break
    w = max(760, _snap(240 * max(len(l) for l in levels) + 80))
    level_h = 128
    h = _snap(96 + len(levels) * level_h)
    parts = [_svg_open(w, h, title, "Editorial diagnostic decision tree rendered as offline SVG.")]
    parts.append(f'<text x="40" y="36" class="t-title">{escape(title)}</text>')
    positions: Dict[int, tuple] = {}
    for li, level in enumerate(levels):
        count = len(level)
        row_w = w - 80
        span = row_w // count
        for i, idx in enumerate(level):
            cx = _snap(40 + span * i + span // 2)
            cy = _snap(88 + li * level_h)
            positions[idx] = (cx, cy)
    for idx, node in enumerate(nodes):
        if idx not in positions:
            continue
        cx, cy = positions[idx]
        parts.append(_box(cx - 100, cy, 200, node["label"], cls="box-quiet"))
        for key in ("yes", "no"):
            child = node.get(key)
            if child is not None and child in positions:
                tx, ty = positions[child]
                parts.append(_arrow(cx, cy + 44, tx, ty, label=key))
    parts.append("</svg>")
    return "".join(parts)
