import re
import warnings
from collections import Counter
from decimal import Decimal
from difflib import SequenceMatcher
from math import exp

import jieba
import numpy as np

with warnings.catch_warnings():
    # jieba.posseg has regex escapes that newer Pythons warn about when compiling it.
    warnings.simplefilter("ignore", SyntaxWarning)
    from jieba import posseg

from ..config import Settings
from ..ingest.chunker import sentence_spans
from ..models import ClaimResult, Evidence, Location, Verdict
from ..retrieval.hybrid import Retriever, cosine_scores
from ..retrieval.tokenize import tokenize
from .nli import NLIModel
from .normalize import normalize

_NUMBER = re.compile(r"(?<![\d.])([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(\s*%)?")
_PERCENT_WORD = re.compile(r"(\d)\s*(?:percent|per cent|pct)\b", re.IGNORECASE)
_PERCENT_PREFIX = re.compile(r"百分之\s*(\d+(?:\.\d+)?)")
_NEGATION_EN = re.compile(
    r"\b(?:not|no|never|without|neither|nor|none|cannot)\b|n['’]t\b", re.IGNORECASE
)
# Matched against whole jieba words, so 不同, 未来, 无人机 and 非常 are not negations.
_NEGATION_ZH = frozenset(
    "不 不必 不能 不会 不得 不可 不要 不再 不用 不曾 不是 没 没有 未 未能 未曾 未经 并未 "
    "并非 从未 从不 无 无法 无须 无需 毫无 非 否 绝不 决不 别 勿".split()
)
_CJK = re.compile(r"[㐀-鿿]+")
_CJK_SPACE = re.compile(r"(?<=[㐀-鿿])\s+|\s+(?=[㐀-鿿])")
_ASCII_WORD = re.compile(r"[a-z0-9]")
_QUALIFIER = re.compile(
    r"\b(?:only|at least|at most|up to|more than|less than|approximately|about|may|might|"
    r"could|expected|projected|proposed|if|unless|estimate\w*)\b|"
    r"至少|至多|不超过|超过|不到|大约|约为|预计|可能|计划|如果|仅",
    re.IGNORECASE,
)
_SURFACE = re.compile(r"[^\W_]+|[%+<>≤≥-]")
_SENTENCE_BREAK = re.compile(r"[.!?;:。！？；：]\s*")
_LATIN_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")
# jieba tags for person, place, organization, and other proper names.
_NAME_FLAGS = frozenset({"nr", "nrfg", "nrt", "ns", "nt", "nz"})
_NUMBERS_ISSUE = "Numbers, dates, or percentages in the claim are missing from the evidence"
_QUALIFIER_ISSUE = "A condition, limit, or uncertainty differs between the claim and the evidence."
# Rule findings that an NLI entailment cannot override; the model tends to miss them.
_BLOCKS_SUPPORT = (_NUMBERS_ISSUE, _QUALIFIER_ISSUE, "Names in the claim")


def _units(text: str) -> str:
    """Normalize typography and write percentages one way: 12 percent and 百分之12 → 12%.

    Spaces next to CJK characters are dropped, so 降低了 8.5% and 降低了8.5% compare equal.
    """
    text = _PERCENT_WORD.sub(r"\1%", normalize(text)[0])
    text = _PERCENT_PREFIX.sub(r"\1%", text)
    return _CJK_SPACE.sub("", text)


def _numbers(text: str) -> Counter[str]:
    values = Counter()
    for match in _NUMBER.finditer(_units(text)):
        value = Decimal(match[1].replace(",", "").lstrip("+"))
        values[format(value.normalize(), "f") + ("%" if match[2] else "")] += 1
    return values


def _negated(text: str) -> bool:
    if _NEGATION_EN.search(text):
        return True
    return any(word in _NEGATION_ZH for run in _CJK.findall(text) for word in jieba.lcut(run))


def _excerpt(claim: str, text: str) -> bool:
    """True when the claim is a word-aligned span of the evidence sentence."""
    claim = _units(claim).casefold().rstrip(".!?;。！？；").strip()
    text = _units(text).casefold()
    start = text.find(claim) if claim else -1
    while start >= 0:
        end = start + len(claim)
        cut_left = start > 0 and _ASCII_WORD.match(claim[0]) and _ASCII_WORD.match(text[start - 1])
        cut_right = (
            end < len(text) and _ASCII_WORD.match(claim[-1]) and _ASCII_WORD.match(text[end])
        )
        if not cut_left and not cut_right:
            return True
        start = text.find(claim, start + 1)
    return False


def _names(text: str) -> set[str]:
    """Proper names and codes: Harbor, BRCA1, IL-6, 青岚. Sentence-initial plain words are skipped."""
    names = set()
    for sentence in _SENTENCE_BREAK.split(text):
        for index, word in enumerate(_LATIN_WORD.findall(sentence)):
            capitals = sum(char.isupper() for char in word)
            code = any(char.isdigit() for char in word) and any(char.isalpha() for char in word)
            if code or capitals > 1 or (capitals and index > 0):
                names.add(word.casefold())
    for run in _CJK.findall(text):
        names.update(
            pair.word for pair in posseg.cut(run) if pair.flag in _NAME_FLAGS and len(pair.word) > 1
        )
    return names


def _missing_names(claim: str, evidence: Evidence) -> list[str]:
    # Names may come from the document title, as in 青岚 for a sentence in 青岚水厂试点.
    context = _units(f"{evidence.location.title} {evidence.text}").casefold()
    return sorted(name for name in _names(_units(claim)) if name not in context)


def _issues(claim: str, text: str) -> list[str]:
    issues = []
    expected, actual = _numbers(claim), _numbers(text)
    if missing := expected - actual:
        issues.append(
            f"{_NUMBERS_ISSUE}: {', '.join(missing.elements())}. "
            f"Evidence has: {', '.join(actual.elements()) or '(none)'}."
        )
    if _negated(claim) != _negated(text):
        issues.append("Negation differs between the claim and the evidence.")
    if set(_QUALIFIER.findall(claim.casefold())) != set(_QUALIFIER.findall(text.casefold())):
        issues.append(_QUALIFIER_ISSUE)
    missing_words = Counter(tokenize(_units(claim))) - Counter(tokenize(_units(text)))
    missing_terms = [
        word for word in missing_words.elements() if not any(char.isdigit() for char in word)
    ]
    if missing_terms:
        issues.append("Claim terms or entities missing from evidence: " + ", ".join(missing_terms))
    return issues


def _candidates(
    retriever: Retriever, claim: str, doc_id: str | None, page: int | None
) -> list[tuple[str, Location]]:
    result = retriever.search(claim, 8, [doc_id] if doc_id is not None else None, page=page)
    anchors = {hit.location.chunk_id: hit.location for hit in result.hits}
    candidates = []
    for text, ids in retriever.store.text_for_chunks(list(anchors)):
        locations = [anchors[chunk_id] for chunk_id in ids]
        for start, end in sentence_spans(text):
            for location in locations:
                if start < location.end and end > location.start:
                    anchor = location.model_copy(update={"start": start, "end": end})
                    candidates.append((text[start:end], anchor))
                    break
    return candidates


def verify_claim(
    retriever: Retriever,
    settings: Settings,
    claim: str,
    doc_id: str | None = None,
    *,
    page: int | None = None,
    nli: NLIModel | None = None,
) -> ClaimResult:
    """Score evidence sentences with lexical rules; with ``nli``, let the NLI model decide."""
    terms = tokenize(_units(claim))
    if not terms:
        return ClaimResult(verdict=Verdict.NOT_FOUND, confidence=0)
    candidates = _candidates(retriever, claim, doc_id, page)
    if not candidates:
        return ClaimResult(verdict=Verdict.NOT_FOUND, confidence=0)
    texts = [text for text, _ in candidates]
    scores = np.zeros(len(texts))
    if retriever.embedder:
        query = retriever.embedder.embed([claim], query=True)
        vectors = retriever.embedder.embed(texts) if query is not None else None
        if vectors is not None:
            scores = cosine_scores(query[0], vectors).clip(0, 1)
    rerank_scores = None
    if settings.claim_rerank and retriever.reranker:
        rerank_scores = retriever.reranker.score(claim, texts)
    ranked = []
    claim_surface = _SURFACE.findall(_units(claim).casefold())
    term_counts = Counter(terms)
    for index, (text, location) in enumerate(candidates):
        words = tokenize(_units(text))
        shared = sum((term_counts & Counter(words)).values())
        coverage = shared / len(terms)
        order = SequenceMatcher(None, terms, words, autojunk=False)
        ordered = sum(block.size for block in order.get_matching_blocks()) / len(terms)
        precision = shared / max(len(words), 1)
        lexical = 0.55 * coverage + 0.35 * ordered + 0.10 * precision
        score = lexical + 0.20 * float(scores[index]) * (1 - lexical)
        if rerank_scores is not None:
            probability = 1 / (1 + exp(-max(-50, min(50, rerank_scores[index]))))
            score += 0.15 * probability * (1 - score)
        issues = _issues(claim, text)
        normalized_text = _units(text)
        # Full support needs the evidence wording: the same sentence or a word-aligned excerpt
        # of it. Paraphrases stay PARTIALLY_SUPPORTED unless the MCP client judges them.
        same_wording = claim_surface == _SURFACE.findall(normalized_text.casefold())
        supports = (
            (same_wording or _excerpt(claim, text))
            and score >= settings.claim_supported_threshold
            and not issues
            and not normalized_text.endswith("?")
        )
        if not supports and ordered < coverage:
            issues.append("The evidence puts matching terms in a different order.")
        verdict = Verdict.NOT_FOUND
        if supports:
            verdict = Verdict.SUPPORTED
        elif score >= settings.claim_partial_threshold and coverage > 0:
            verdict = Verdict.PARTIALLY_SUPPORTED
        ranked.append((verdict, Evidence(text=text, location=location, score=score), issues))
    in_order = list(ranked)  # Document order, before ranking.
    by_score = sorted(ranked, key=lambda item: -item[1].score)
    priority = {Verdict.SUPPORTED: 2, Verdict.PARTIALLY_SUPPORTED: 1, Verdict.NOT_FOUND: 0}
    ranked.sort(key=lambda item: (priority[item[0]], item[1].score, -len(item[2])), reverse=True)
    verdict, best, issues = ranked[0]
    result = ClaimResult(
        verdict=verdict,
        confidence=best.score,
        evidence=[item[1] for item in ranked[:3]],
        issues=issues,
    )
    if nli is None:
        return result
    top = by_score[: settings.nli_candidates]
    premises = top + _windows(claim, in_order, {id(item) for item in top})
    return _judge_with_nli(nli, settings, claim, premises, result)


def _windows(
    claim: str, in_order: list[tuple[Verdict, Evidence, list[str]]], selected: set[int]
) -> list[tuple[Verdict, Evidence, list[str]]]:
    """Join each selected sentence with its neighbor, since evidence often spans two sentences.

    On 250 SciFact train pairs, adding these windows raised NLI macro-F1 from 0.43 to 0.48
    without adding false SUPPORT verdicts at the default threshold.
    """
    windows = []
    for first, second in zip(in_order, in_order[1:], strict=False):
        a, b = first[1], second[1]
        adjacent = a.location.doc_id == b.location.doc_id and a.location.end <= b.location.start
        if adjacent and (id(first) in selected or id(second) in selected):
            text = f"{a.text} {b.text}"
            location = a.location.model_copy(update={"end": b.location.end})
            evidence = Evidence(text=text, location=location, score=max(a.score, b.score))
            windows.append((Verdict.NOT_FOUND, evidence, _issues(claim, text)))
    return windows


def _judge_with_nli(
    nli: NLIModel,
    settings: Settings,
    claim: str,
    candidates: list[tuple[Verdict, Evidence, list[str]]],
    rules: ClaimResult,
) -> ClaimResult:
    """Decide with entailment probabilities over the best rule-ranked evidence sentences."""
    probabilities = nli.predict([(evidence.text, claim) for _, evidence, _ in candidates])
    if probabilities is None:
        rules.issues.append("NLI model unavailable. Used rule-based verification.")
        return rules
    entail = max(range(len(candidates)), key=lambda i: probabilities[i]["entailment"])
    contra = max(range(len(candidates)), key=lambda i: probabilities[i]["contradiction"])
    p_entail = probabilities[entail]["entailment"]
    p_contra = probabilities[contra]["contradiction"]

    if p_contra >= settings.nli_contradict_threshold and p_contra > p_entail:
        lead, verdict, confidence = contra, Verdict.CONTRADICTED, p_contra
        issues = [*candidates[lead][2], f"NLI contradiction probability {p_contra:.2f}."]
    elif rules.verdict == Verdict.SUPPORTED:
        # Matching wording is stronger evidence than the model; keep the rule verdict.
        return rules.model_copy(update={"judged_by": "nli"})
    elif p_entail >= settings.nli_entail_threshold:
        # The model often accepts a changed figure, a dropped "up to", or a swapped name, so
        # support needs an entailing sentence without those rule findings.
        blocking = {}
        for i, (_, evidence, found) in enumerate(candidates):
            if probabilities[i]["entailment"] >= settings.nli_entail_threshold:
                names = _missing_names(claim, evidence)
                if names:
                    found = [
                        *found,
                        f"Names in the claim are missing from the evidence: {', '.join(names)}.",
                    ]
                blocking[i] = [issue for issue in found if issue.startswith(_BLOCKS_SUPPORT)]
        clean = [i for i, found in blocking.items() if not found]
        lead = max(clean or blocking, key=lambda i: probabilities[i]["entailment"])
        confidence = probabilities[lead]["entailment"]
        verdict = Verdict.SUPPORTED if clean else Verdict.PARTIALLY_SUPPORTED
        issues = [*blocking[lead], f"NLI entailment probability {confidence:.2f}."]
    else:
        lead = entail
        verdict = rules.verdict
        confidence = probabilities[lead]["neutral"]
        issues = [*rules.issues, "NLI found no sentence that entails or contradicts the claim."]
    evidence = [candidates[lead][1]] + [item[1] for i, item in enumerate(candidates) if i != lead][
        :2
    ]
    return ClaimResult(
        verdict=verdict, confidence=confidence, evidence=evidence, issues=issues, judged_by="nli"
    )
