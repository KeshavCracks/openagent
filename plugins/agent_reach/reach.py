"""Sandboxed, read-only public medical grounding.

Boundaries (locked):
  * ONLY lawful, public research lookup: PubMed E-utilities, WHO/CDC RSS,
    lecture-transcript cleanup. Opt-in via Settings (default OFF).
  * Allowlist enforcement: requests to hosts outside ALLOWED_HOSTS are
    refused before any socket opens. HTTP verbs: GET only.
  * Privacy: request payloads contain ONLY the query terms — never session
    data, never chat history (enforced by `validate_pre_action('external_network')`
    in the quality gates, and by the shape of the functions below).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional
from urllib.parse import urlencode, urlparse
from xml.etree import ElementTree

import httpx

ALLOWED_HOSTS = frozenset(
    {
        "eutils.ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "www.who.int",
        "www.cdc.gov",
    }
)
_TIMEOUT = 10.0
_UA = "MedHarness-Grounding/1.0 (public literature lookup; read-only)"


def _assert_allowed(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"Blocked non-allowlisted host: {parsed.hostname!r}")
    return url


# ------------------------------------------------------------- PubMed ------
def pubmed_search(query: str, retmax: int = 5) -> Dict[str, object]:
    """esearch + esummary for public literature. Returns {results|warning}."""
    if not query.strip():
        return {"results": [], "warning": "empty query"}
    retmax = max(1, min(int(retmax), 20))
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _UA}) as client:
            url = _assert_allowed(f"{base}/esearch.fcgi?{urlencode({'db': 'pubmed', 'term': query, 'retmode': 'json', 'retmax': retmax})}")
            ids = client.get(url).json()["esearchresult"]["idlist"]
            if not ids:
                return {"results": [], "warning": None}
            url = _assert_allowed(f"{base}/esummary.fcgi?{urlencode({'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'json'})}")
            summary = client.get(url).json()["result"]
            results = [
                {
                    "pmid": uid,
                    "title": summary[uid].get("title", ""),
                    "journal": summary[uid].get("source", ""),
                    "year": summary[uid].get("pubdate", "")[:4],
                }
                for uid in ids
                if uid in summary
            ]
            return {"results": results, "warning": None}
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        return {"results": [], "warning": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------- RSS ------
def fetch_rss_headlines(url: str, limit: int = 8) -> Dict[str, object]:
    """Read-only Atom/RSS ingest (WHO/CDC newsroom feeds)."""
    limit = max(1, min(int(limit), 20))
    try:
        _assert_allowed(url)
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _UA}, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
        items: List[Dict[str, str]] = []
        # Atom
        for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
            title = entry.findtext("{http://www.w3.org/2005/Atom}title", "").strip()
            link_el = entry.find("{http://www.w3.org/2005/Atom}link")
            items.append({"title": title, "link": link_el.get("href", "") if link_el is not None else ""})
            if len(items) >= limit:
                return {"results": items, "warning": None}
        # RSS 2.0
        for item in root.iter("item"):
            items.append({"title": (item.findtext("title") or "").strip(), "link": (item.findtext("link") or "").strip()})
            if len(items) >= limit:
                break
        return {"results": items, "warning": None}
    except (httpx.HTTPError, ElementTree.ParseError, ValueError) as exc:
        return {"results": [], "warning": f"{type(exc).__name__}: {exc}"}


# ----------------------------------------------------------- transcripts ---
_TS_STAMP = re.compile(r"^\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*")
_INLINE_STAMP = re.compile(r"(?:\d{1,2}:){1,2}\d{2}")
_SPEAKER_MARK = re.compile(r"^>>\s*")


def parse_transcript(text: str) -> str:
    """Strip timestamps + speaker marks from lecture transcripts, collapse repeats."""
    lines: List[str] = []
    last: Optional[str] = None
    for raw in text.splitlines():
        line = _SPEAKER_MARK.sub("", raw.strip())
        line = _TS_STAMP.sub("", line)
        line = _INLINE_STAMP.sub("", line).strip()
        if not line or line == last:
            continue
        lines.append(line)
        last = line
    return "\n".join(lines)
