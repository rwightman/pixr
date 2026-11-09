from typing import Dict, List, Sequence

try:
    import Levenshtein
except ImportError:
    Levenshtein = None


def _levenshtein(s1: str, s2: str) -> int:
    if Levenshtein is not None:
        return Levenshtein.distance(s1, s2)
    # simple DP fallback
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        cur = [i]
        for j, c2 in enumerate(s2, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (c1 != c2)))
        prev = cur
    return prev[-1]


def normalized_levenshtein(s1: str, s2: str) -> float:
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.
    return _levenshtein(s1, s2) / max_len


def anls_score(prediction: str, answers: Sequence[str], tau: float = 0.5) -> float:
    """ Average Normalized Levenshtein Similarity for one question (as per DocVQA eval).

    Comparison is case-insensitive w/ surrounding whitespace removed.
    """
    pred = ' '.join(prediction.strip().lower().split())
    best = 0.
    for a in answers:
        a = ' '.join(a.strip().lower().split())
        nl = normalized_levenshtein(pred, a)
        best = max(best, 1. - nl if nl < tau else 0.)
    return best


def average_normalized_levenshtein_similarity(
        ground_truth: Sequence[Sequence[str]],
        predicted_answers: Sequence[str],
        tau: float = 0.5,
) -> float:
    assert len(ground_truth) == len(predicted_answers), "Length of ground_truth and predicted_answers must match."
    if not ground_truth:
        return 0.
    return sum(anls_score(p, a, tau) for a, p in zip(ground_truth, predicted_answers)) / len(ground_truth)


def text_error_rates(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """ Corpus level character & word error rates. """
    char_errors = sum(_levenshtein(p, r) for p, r in zip(predictions, references))
    char_total = sum(len(r) for r in references)
    word_errors = sum(_levenshtein(p.split(), r.split()) for p, r in zip(predictions, references))
    word_total = sum(len(r.split()) for r in references)
    return {
        'cer': char_errors / max(char_total, 1),
        'wer': word_errors / max(word_total, 1),
    }
