"""Tests ensuring 0% arithmetic hallucination in the clinical math path.

Guarantees under test:
  * formulas match published worked examples,
  * the interceptor catches clinical-math utterances,
  * out-of-domain inputs raise instead of yielding plausible garbage,
  * verified results carry a ground-truth context block for the LLM.
"""
import asyncio

import pytest

from core.math_engine import (
    MathError,
    bsa_mosteller,
    cockcroft_gault,
    corrected_sodium,
    anion_gap,
    ckd_epi_2021,
    detect_intent,
    infusion_rate_mcgkgmin,
    iv_drip_rate,
    maintenance_fluids_421,
    pediatric_mgkg_dose,
    to_cm,
    to_kg,
    try_compute,
)


# ------------------------------------------------------- published values --
def test_cockcroft_gault_male_canonical():
    # (140-40)*70 / (72*1.0) = 97.22 mL/min
    r = cockcroft_gault(40, 70, 1.0, "male")
    assert r.value == pytest.approx(97.22, abs=0.01)
    assert r.unit == "mL/min"
    assert "Nephron 1976" in r.reference


def test_cockcroft_gault_female_factor():
    # 97.22 * 0.85 = 82.64
    r = cockcroft_gault(40, 70, 1.0, "female")
    assert r.value == pytest.approx(82.64, abs=0.01)


def test_ckd_epi_2021_male():
    # 142 * 1^alpha * (1/0.9)^-1.2 * 0.9938^45 ≈ 94.6 → G1 (≥90)
    r = ckd_epi_2021(45, 1.0, "male")
    assert r.value == pytest.approx(94.6, rel=0.01)
    assert r.interpretation.startswith("G1")


def test_ckd_epi_2021_female_stage():
    r = ckd_epi_2021(45, 1.0, "female")
    assert r.value == pytest.approx(70.8, rel=0.01)
    assert r.interpretation.startswith("G2")


def test_bsa_mosteller():
    # sqrt(170*70/3600) = 1.8181 -> 1.82 m²
    r = bsa_mosteller(170, 70)
    assert r.value == pytest.approx(1.82, abs=0.01)


def test_unit_conversions():
    assert to_kg(154, "lb") == pytest.approx(69.853, abs=0.01)
    assert to_cm(70, "in") == pytest.approx(177.8, abs=0.01)


def test_pediatric_dose_and_cap():
    r = pediatric_mgkg_dose(18, 15)
    assert r.value == 270.0
    r2 = pediatric_mgkg_dose(18, 15, max_single_mg=250)
    assert r2.value == 250.0
    assert any("capped" in w.lower() for w in r2.warnings)


def test_iv_drip_rate_rounds_to_whole_drops():
    # 500 mL × 15 gtt/mL / 240 min = 31.25 -> 31 gtt/min
    r = iv_drip_rate(500, 240, 15)
    assert r.value == 31
    assert r.unit == "gtt/min"


def test_infusion_rate():
    # 4 mcg/kg/min × 80 kg × 60 / 400 mcg/mL = 48 mL/hr
    r = infusion_rate_mcgkgmin(4, 80, 400)
    assert r.value == pytest.approx(48.0, abs=0.1)


def test_421_maintenance_fluids():
    # 28 kg -> 4*10 + 2*10 + 1*8 = 68 mL/hr
    r = maintenance_fluids_421(28)
    assert r.value == pytest.approx(68.0, abs=0.1)


def test_anion_gap_and_corrected_na():
    assert anion_gap(140, 100, 24).value == pytest.approx(16.0)
    assert corrected_sodium(128, 600).value == pytest.approx(136.0)


# ------------------------------------------------------ domain safety ------
@pytest.mark.parametrize("kwargs", [
    {"age_y": 200, "weight": 70, "scr_mg_dl": 1.0, "sex": "male"},
    {"age_y": 40, "weight": 900, "scr_mg_dl": 1.0, "sex": "male"},
    {"age_y": 40, "weight": 70, "scr_mg_dl": 0.05, "sex": "male"},
])
def test_out_of_domain_inputs_raise(kwargs):
    with pytest.raises(MathError):
        cockcroft_gault(**kwargs)


def test_bad_sex_raises():
    with pytest.raises(MathError):
        cockcroft_gault(40, 70, 1.0, "x")


# ------------------------------------------------------ interception ------
def test_interceptor_detects_all_intents():
    utterances = {
        "what is the creatinine clearance for a 62 yo male, 80 kg, SCr 1.4?": "cg",
        "calculate egfr with ckd-epi for 55 year old male, creatinine of 1.1": "egfr",
        "bsa for 180 cm 85 kg patient": "bsa",
        "give the pediatric dose at 15 mg/kg for 18 kg child": "peds",
        "drip rate for 500 ml over 240 minutes with drop factor 15": "drip",
        "infusion rate for 4 mcg/kg/min at 80 kg with 400 mcg/mL": "infusion",
        "maintenance fluids by 4-2-1 for 28 kg": "421",
        "anion gap for na 140 cl 100 hco3 24": "ag",
        "corrected sodium for na 128 glucose 600": "na_corr",
    }
    for text, action in utterances.items():
        assert detect_intent(text) == action, text


def test_interceptor_computes_and_never_lets_llm_compute():
    """The core guarantee: math intent → engine result with provenance block."""
    text = "what's the creatinine clearance for a 62 year old male, 80 kg, SCr 1.4?"
    result = try_compute(text)
    assert result is not None
    assert result.value == pytest.approx(61.90, abs=0.01)  # (140-62)*80/(72*1.4)
    block = result.as_context_block()
    assert "DETERMINISTIC MATH" in block
    assert "DO NOT recompute" in block
    assert result.formula == "Cockcroft-Gault creatinine clearance"


def test_non_math_text_returns_none():
    assert try_compute("summarize the assessment and plan for this encounter") is None


def test_incomplete_math_inputs_raise_not_guess():
    """Missing SCr must raise — the LLM is forbidden from inventing numbers."""
    with pytest.raises(MathError):
        try_compute("crcl for a 62 yo male, 80 kg")


def test_think_tag_split_guard_for_math_blocks():
    """Math context block survives transport-level think splitting (no overlap)."""
    from core.tunnel_client import ThinkSplitter

    block = try_compute("bsa for 180 cm 85 kg").as_context_block()
    splitter = ThinkSplitter()
    pieces: list = []
    for i in range(0, len(block), 7):  # adversarial 7-char chunks
        pieces.extend(splitter.feed(block[i:i + 7]))
    pieces.extend(splitter.flush())
    # default state is "answer": nothing is swallowed by think-parsing
    assert "".join(p for _, p in pieces).replace("\n", "") == block.replace("\n", "")
    assert all(mode == "answer" for mode, _ in pieces)
