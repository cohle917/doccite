from collections import Counter

import pytest

from doccite.models import Verdict
from doccite.retrieval.hybrid import Retriever
from doccite.verify.claim import _excerpt, _negated, _numbers, verify_claim
from doccite.verify.normalize import normalize
from doccite.verify.quote import verify_quote


def test_normalization_maps_to_extracted_text():
    text = "  “Ａ\n\tCafe\u0301”  "
    canonical, offsets = normalize(text)
    assert canonical == '"A Café"'
    assert text[slice(*offsets[2])] == "\n\t"
    assert text[slice(*offsets[6])] == "e\u0301"


def test_quote_verbatim_normalized_modified_and_missing(store, indexer, pdf_path):
    doc = indexer.file(pdf_path).document
    actual = "The pilot reduced operating costs by 12%."
    result = verify_quote(store, actual, doc.doc_id, 0.8)
    assert result.status == "verbatim"
    assert result.match.location.page == 1
    assert result.differences == []
    normalized = verify_quote(store, actual.replace("12%", "１２％"), doc.doc_id, 0.8)
    assert normalized.status == "normalized"
    changed = verify_quote(store, actual.replace("12%", "21%"), doc.doc_id, 0.8)
    assert changed.status == "modified"
    assert changed.match.text == actual
    assert changed.differences
    assert verify_quote(store, "Remote satellite", doc.doc_id, 0.8).status == "not_found"
    assert verify_quote(store, actual, "fabricated", 0.8).status == "not_found"


@pytest.mark.parametrize(
    ("claim", "verdict", "issue"),
    [
        ("The pilot reduced operating costs by 12%.", Verdict.SUPPORTED, None),
        ("The pilot reduced operating costs by 21%.", Verdict.PARTIALLY_SUPPORTED, "21%"),
        ("The pilot reduced operating costs by 12 percent.", Verdict.SUPPORTED, None),
        # The evidence sentence also carries 2025; extra evidence numbers are not a mismatch.
        ("Alice approved the final plan.", Verdict.SUPPORTED, None),
        ("Alice approved the final plan in 2024.", Verdict.PARTIALLY_SUPPORTED, "2024"),
        (
            "The pilot did not reduce operating costs by 12%.",
            Verdict.PARTIALLY_SUPPORTED,
            "Negation",
        ),
        ("Bob approved the final plan in 2025.", Verdict.PARTIALLY_SUPPORTED, "entities"),
        ("Neptune has visible rings.", Verdict.NOT_FOUND, None),
    ],
)
def test_claim_checks_numbers_negation_entities_and_absence(
    store, settings, indexer, pdf_path, claim, verdict, issue
):
    doc = indexer.file(pdf_path).document
    result = verify_claim(Retriever(store, None, None), settings, claim, doc.doc_id)
    assert result.verdict == verdict
    if issue:
        assert any(issue in problem for problem in result.issues)


def test_numbers_compare_values_and_percent_spellings():
    assert _numbers("百分之 8.5, 8.50 percent and 1,200 in 2025年") == Counter(
        {"8.5%": 2, "1200": 1, "2025": 1}
    )
    assert not _numbers("by 12%") - _numbers("In June 2025 costs fell by 12%.")
    assert _numbers("by 21%") - _numbers("costs fell by 12%")


@pytest.mark.parametrize(
    ("text", "negated"),
    [
        ("巡检团队使用无人机检查了塔架。", False),
        ("未来计划扩展到 60 座。", False),
        ("两组用水量不同。", False),
        ("温室环境非常稳定。", False),
        ("样品不合格。", True),
        ("检修前不必断开进水泵电源。", True),
        ("值班人员未采集水样。", True),
        ("The volunteers did not remove samples.", True),
        ("The volunteers didn't remove samples.", True),
        ("Nothing was removed from the notebook.", False),
    ],
)
def test_negation_uses_whole_words(text, negated):
    assert _negated(text) is negated


def test_excerpt_requires_word_aligned_span():
    evidence = "The lockers use a six-digit pickup code that expires after 72 hours."
    assert _excerpt("The lockers use a six-digit pickup code.", evidence)
    assert not _excerpt("The lockers use a six-digit pick", evidence)
    assert not _excerpt("safe zone", "An unsafe zone was marked.")
    assert _excerpt("共采集 36 份出厂水样。", "2025 年 4 月共采集 36 份出厂水样。")
    assert _excerpt("costs fell by 12 percent", "Costs fell by 12% in June.")
    assert _excerpt("能耗降低了百分之 8.5。", "试点将单位产水能耗降低了 8.5%。")
