"""v1.1 feature tests — new calculators, model-health panel, evidence endpoint.

Offline guarantees:
  * calculators are anchored to canonical published worked examples;
  * HTTP endpoints run against the FastAPI app with a stubbed tunnel and a
    monkeypatched PubMed fetcher — no sockets are ever opened;
  * settings/state are isolated per test via MEDHARNESS_HOME (tmp_path).
"""
import pytest
from fastapi.testclient import TestClient

import app.main as appmain
from core.math_engine import (  # noqa: F401
    MathError,
    cha2ds2_vasc,
    corrected_calcium,
    curb65,
    detect_intent,
    mean_arterial_pressure,
    try_compute,
    wells_dvt,
    wells_pe,
)


# ------------------------------------------------------------- MAP ----
def test_map_canonical():
    r = mean_arterial_pressure(120, 80)
    assert r.value == pytest.approx(93.3, abs=0.05)
    assert r.unit == "mmHg"
    assert "adequate" in r.interpretation


def test_map_borderline_and_shock_warning():
    r = mean_arterial_pressure(88, 54)
    assert r.value == pytest.approx(65.3, abs=0.05)
    assert "borderline" in r.interpretation
    r2 = mean_arterial_pressure(78, 40)
    assert any("shock" in w for w in r2.warnings)


def test_map_rejects_inverted_bp():
    with pytest.raises(MathError):
        mean_arterial_pressure(80, 90)


# ------------------------------------------------- corrected calcium ----
def test_corrected_calcium_mg_dl():
    r = corrected_calcium(7.6, 2.8, "mg/dl", "g/dl")
    assert r.value == pytest.approx(8.56, abs=0.01)
    assert r.unit == "mg/dL"


def test_corrected_calcium_si():
    r = corrected_calcium(1.9, 28, "mmol/l", "g/l")
    assert r.value == pytest.approx(2.14, abs=0.01)
    assert "Within normal" in r.interpretation


def test_corrected_calcium_flags_low():
    r = corrected_calcium(6.0, 2.0, "mg/dl", "g/dl")
    assert r.value == pytest.approx(7.6, abs=0.01)
    assert "hypocalcemia" in r.interpretation


# ----------------------------------------------------- CHA2DS2-VASc ----
def test_chads_published_weighting():
    r = cha2ds2_vasc(72, "male", "HTN, DM, prior stroke")  # 1+1+2, no CHF/vascular
    assert r.value == 5
    assert "High" in r.interpretation


def test_chads_negation_respected():
    r = cha2ds2_vasc(74, "male", "hypertension, no diabetes, no prior stroke, no heart failure")
    assert r.value == 2  # HTN + age only
    assert "High" in r.interpretation


def test_chads_female_sex_alone_is_low():
    r = cha2ds2_vasc(34, "female", "no comorbidities")
    assert r.value == 1
    assert "Low" in r.interpretation


# ------------------------------------------------------------ Wells ----
def test_wells_dvt_score_and_band():
    r = wells_dvt("active cancer on chemo, calf swelling 4 cm, tenderness along deep vein; "
                  "alternative diagnosis less likely")
    assert r.value == 1 + 1 + 1 - 2
    assert "moderate" in r.interpretation.lower()


def test_wells_dvt_refuses_empty_chart():
    with pytest.raises(MathError):
        wells_dvt("no findings documented")


def test_wells_pe_half_points():
    r = wells_pe("HR 110 bpm, hemoptysis, known malignancy")
    assert r.value == 3.5
    assert "moderate" in r.interpretation.lower()


# ---------------------------------------------------------- CURB-65 ----
def test_curb65_full_criteria():
    r = curb65(72, "confused, RR 32, BP 85/50, urea 9 mmol/L")
    assert r.value == 5
    assert "severe" in r.interpretation.lower() or "hospital" in r.interpretation.lower()


def test_curb65_warns_undocumented():
    r = curb65(40, "")
    assert r.value == 0
    assert any("underestimate" in w for w in r.warnings)


# ------------------------------------------------------ interception ----
def test_interceptor_routes_new_intents():
    assert detect_intent("what's the MAP given BP 120/80?") == "map"
    assert detect_intent("albumin-corrected calcium please") == "ca_corr"
    assert detect_intent("CHADS-VASc in afib") == "chads"
    assert detect_intent("wells criteria for DVT") == "wells"
    assert detect_intent("CURB-65 score") == "curb65"
    # innocent words must NOT trigger interception
    assert detect_intent("the roadmap for discharge planning") is None
    assert detect_intent("map out the differential without vitals") is None


def test_try_compute_end_to_end_new_paths():
    r = try_compute("BP 120/80, calculate the map")
    assert r.value == pytest.approx(93.3, abs=0.05)
    r = try_compute("CHADS-VASc for 72yo male with hypertension, diabetes, prior stroke")
    assert r.value == 5
    assert "DO NOT recompute" in r.as_context_block()


# ---------------------------------------------------------- endpoints ----
class _StubTunnel:
    def __init__(self, ids):
        self._ids = ids

    async def list_models(self):
        return {"data": [{"id": i} for i in self._ids]}

    async def aclose(self):  # shutdown handler contract
        pass


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDHARNESS_HOME", str(tmp_path))
    monkeypatch.setattr(appmain, "_state", {"tunnel": None, "heartbeat": None, "models_cache": None})
    import app.config as cfg
    monkeypatch.setattr(cfg, "_settings", None)
    with TestClient(appmain.app) as c:
        yield c


def test_models_endpoint_without_tunnel(client):
    body = client.get("/api/models").json()
    assert body["tunnel_configured"] is False
    assert body["models"] == []


def test_models_endpoint_maps_aliases(client):
    cfg = get_settings_silent()
    cfg.remote_base_url = "https://example.trycloudflare.com"
    appmain._state["tunnel"] = _StubTunnel(
        ["iqvia-ai/Med-R1-8B-GPTQ-W4A16", "google/medgemma-4b-it-gptq-w4a16"])
    models = {m["alias"]: m["served"] for m in client.get("/api/models").json()["models"]}
    assert models == {"medr1": True, "medgemma": True}


def test_models_endpoint_survives_tunnel_failure(client):
    cfg = get_settings_silent()
    cfg.remote_base_url = "https://example.trycloudflare.com"

    class _Bad:
        async def list_models(self):
            raise RuntimeError("boom")

        async def aclose(self):
            pass

    appmain._state["tunnel"] = _Bad()
    body = client.get("/api/models").json()
    assert body["tunnel_configured"] is True
    assert body["models"] == []
    assert "RuntimeError" in (body["warning"] or "")


def test_evidence_requires_opt_in(client):
    assert client.post("/api/evidence", json={"query": "sepsis bundles"}).status_code == 403


def test_evidence_returns_pubmed_results(client, monkeypatch):
    cfg = get_settings_silent()
    cfg.agent_reach_enabled = True

    def fake_pubmed(query, retmax=5):  # sync — main runs it via asyncio.to_thread
        return {"results": [{"pmid": "12345", "title": "Early SEPSIS bundle",
                             "journal": "NEJM", "year": "2024"}], "warning": None}

    monkeypatch.setattr(appmain, "pubmed_search", fake_pubmed)
    body = client.post("/api/evidence", json={"query": "sepsis bundle mortality"}).json()
    assert body["results"][0]["url"].endswith("/12345/")
    assert "PubMed" in body["source"]


def get_settings_silent():
    """Fresh settings singleton after the fixture reset (no file writes beyond tmp)."""
    from app.config import get_settings
    return get_settings()
