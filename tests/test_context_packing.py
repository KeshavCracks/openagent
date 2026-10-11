"""Tests measuring token-reduction efficiency of the local context packer.

The remote T4's prefill time is proportional to uploaded tokens, so the
packer must provably cut boilerplate without losing decisive clinical
content, and must report honest stats.
"""
from plugins.context_optimizer.optimizer import chunk, pack, pack_message

SAMPLE_EHR = """CONFIDENTIALITY NOTICE: This message is intended only for the addressee.
Privileged and confidential. If you are not the addressee, please delete.
Page 1 of 6
 Printed on 2026-10-11 by Ward Terminal 3

HISTORY OF PRESENT ILLNESS
62 year old male admitted with progressive dyspnea and bilateral leg swelling.
Orthopnea to two pillows. No chest pain. Prior MI 2019.

MEDICATIONS
Aspirin 81 mg daily
Atorvastatin 40 mg nightly
Metoprolol succinate 50 mg daily

CONFIDENTIALITY NOTICE: This message is intended only for the addressee.
Page 2 of 6

LABS
Na 138, K 4.2, Cl 100, HCO3 24, BUN 28, SCr 1.4
BNP 1240
Hb 11.9, WBC 7.2, Plt 220

LABS
Na 138, K 4.2, Cl 100, HCO3 24, BUN 28, SCr 1.4
Page 3 of 6

ASSESSMENT
Decompensated heart failure, likely HFrEF given prior MI.

CONFIDENTIALITY NOTICE: This message is intended only for the addressee.
"""


def test_boilerplate_and_page_furniture_removed():
    ctx = pack(SAMPLE_EHR)
    lowered = ctx.text.lower()
    assert "confidentiality notice" not in lowered
    assert "page 3 of 6" not in lowered
    assert "printed on" not in lowered


def test_duplicate_lab_block_deduplicated():
    ctx = pack(SAMPLE_EHR)
    # the identical LABS line pair appears twice in the raw sample
    raw_pairs = SAMPLE_EHR.count("Na 138, K 4.2, Cl 100, HCO3 24, BUN 28, SCr 1.4")
    packed_pairs = ctx.text.count("Na 138, K 4.2, Cl 100, HCO3 24, BUN 28, SCr 1.4")
    assert raw_pairs == 2
    assert packed_pairs == 1


def test_decisive_content_survives_packing():
    ctx = pack(SAMPLE_EHR)
    for fact in ("Orthopnea to two pillows", "Metoprolol succinate 50 mg", "BNP 1240", "HFrEF"):
        assert fact in ctx.text, f"decisive fact lost: {fact}"


def test_token_reduction_target():
    """Boilerplate-heavy EHR dumps must cut ≥25% of tokens (TTFT win)."""
    ctx = pack(SAMPLE_EHR)
    assert ctx.saved_pct >= 25.0, ctx.stats()
    stats = ctx.stats()
    assert stats["packed_tokens"] < stats["original_tokens"]
    assert stats["removed_lines"] >= 6


def test_stats_are_consistent():
    ctx = pack(SAMPLE_EHR)
    assert ctx.packed_chars == len(ctx.text)
    assert ctx.original_chars == len(SAMPLE_EHR)
    assert ctx.packed_tokens == len(ctx.text) // 4


def test_chunking_respects_budget_and_labels():
    text = ("Paragraph about cardiology. " * 60) + "\n\n" + ("Second long paragraph. " * 60)
    chunks = chunk(text, chunk_chars=800)
    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c.startswith(f"[CHUNK {i + 1}/{len(chunks)}]")
        # label excluded from the payload budget
        payload = c.split("]\n", 1)[1]
        assert len(payload) <= 800 + 40  # small paragraph-boundary tolerance


def test_empty_and_clean_input():
    empty = pack("")
    assert empty.packed_chars == 0 and empty.saved_pct == 0.0
    clean = pack("Short clean note already free of boilerplate.")
    assert clean.saved_pct == 0.0  # packer must never expand clinical text


def test_pack_message_wrapper_shape():
    text, stats = pack_message(SAMPLE_EHR)
    assert isinstance(stats, dict)
    assert set(stats) == {"original_tokens", "packed_tokens", "saved_pct", "chunks", "removed_lines"}
    assert text  # non-empty packing
