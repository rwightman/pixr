from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

try:
    from rapidfuzz.distance import Levenshtein as _RfLevenshtein
    from rapidfuzz.process import cdist as _rf_cdist
except ImportError:
    _RfLevenshtein = _rf_cdist = None


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


# --- line matched CER (lm_cer), reading order robust ---
# lm_cer forgives reading order differences (columns, blocks, form fields read in another reasonable order) but still
# charges misreads, omissions, hallucination and repetition loops in full. Per page edits are min(CER edits, line
# matched edits), so lm_cer <= CER on every page w/ the same denominator, and lm_cer_gap = CER - lm_cer is the share
# of CER from reading order / line segmentation differences.
# Line matching, every line carrying its line break (an unpaired line costs len + 1, a pair costs lev(p, r)):
#   1. identical lines are paired first (multiset intersection, always part of an optimal pairing, and repeats beyond
#      the reference's own count stay unpaired, so loops are charged)
#   2. optimal 1:1 assignment over the remaining lines, restricted to pairs where one line roughly contains the other:
#      lev(p, r) - |len p - len r| <= gate * min(len p, len r)
#   3. split / merge repair: an unpaired line is joined ('\n') to either end of a paired group's pred or ref side when
#      that lowers the group's edits by >= min_explained * len(line) (greedy, best gain first), so a line the model
#      split or merged costs ~1 edit as CER would w/ the same order
#   4. leftover lines: unrestricted optimal 1:1 assignment (substitutions), the rest are insertions / deletions
# order_tau: mean per page Kendall tau between the predicted and reference order of confidently paired lines.
# Based on the line matched measure of Leifert et al., "End-To-End Measure for Text Recognition", ICDAR 2019
# (arXiv 1908.09584: min cost 1:1 line assignment + unmatched lengths, w/ a split / merge variant), w/ the order score
# reported separately as in Vidal et al., "End-to-End Page-Level Assessment of Handwritten Text Recognition", Pattern
# Recognition 2023 (arXiv 2301.05935). Stages 1 / 2 gating, the split / merge repair, the per page CER cap and the
# Kendall tau order score are pixr specific, values aren't comparable to either paper's. Known blind spot: a table
# read column-wise instead of row-wise gets lm_cer ~0, only order_tau shows it.

def linear_sum_assignment(cost) -> Tuple[np.ndarray, np.ndarray]:
    """ Min cost rectangular assignment (shortest augmenting path w/ potentials), numpy only.

    Same result (optimal cost, rows sorted) as scipy.optimize.linear_sum_assignment for finite costs, w/o the scipy
    dependency. O(n^2 m) w/ a vectorized inner loop, fast enough for per page line matching.
    """
    cost = np.asarray(cost, dtype=np.float64)
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape  # n <= m, every row gets a column
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    col_row = np.zeros(m + 1, dtype=np.int64)  # 1-based row assigned to column j (0 = free), column 0 is the root
    for i in range(1, n + 1):
        col_row[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        way = np.zeros(m + 1, dtype=np.int64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = col_row[j0]
            cur = cost[i0 - 1] - u[i0] - v[1:]  # reduced costs from row i0
            free = ~used[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            masked = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(masked)) + 1
            delta = masked[j1 - 1]
            u[col_row[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if col_row[j0] == 0:
                break
        while j0:  # augment along the alternating path
            j1 = way[j0]
            col_row[j0] = col_row[j1]
            j0 = j1
    cols = np.nonzero(col_row[1:])[0]
    rows = col_row[1:][cols] - 1
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]


def _kendall_tau(a: Sequence[int], b: Sequence[int]) -> float:
    """ Kendall tau-a (== tau-b w/o ties, as for 1:1 paired line indices). """
    a, b = np.asarray(a), np.asarray(b)
    s = np.sign(a[:, None] - a[None]) * np.sign(b[:, None] - b[None])
    return float(s.sum() / (len(a) * (len(a) - 1)))


def _assign_lines(P: Sequence[str], R: Sequence[str], gate: Optional[float] = None) -> List[Tuple[int, int, int]]:
    """ Optimal 1:1 line pairs (i, j, lev) minimising pair edits + unpaired (len + 1), gate restricts the pairs. """
    if not P or not R:
        return []
    D = _rf_cdist(P, R, scorer=_RfLevenshtein.distance, dtype=np.int32, workers=1)
    lp = np.array([len(p) for p in P])[:, None]
    lr = np.array([len(r) for r in R])[None, :]
    ok = np.ones_like(D, dtype=bool)
    if gate is not None:
        ok = ((D - np.abs(lp - lr)) <= gate * np.minimum(lp, lr)) & (np.minimum(lp, lr) > 0)
    # lines w/o any allowed pair (all zero savings) can't change the optimum, solve on the rest
    ri, ci = np.nonzero(ok.any(1))[0], np.nonzero(ok.any(0))[0]
    if not len(ri):
        return []
    # savings vs leaving both unpaired, <= 0 as lev(p, r) <= len p + len r
    savings = np.where(ok, D - (lp + 1) - (lr + 1), 0)[np.ix_(ri, ci)]
    rows, cols = linear_sum_assignment(savings)
    rows, cols = ri[rows], ci[cols]
    return [(int(a), int(b), int(D[a, b])) for a, b in zip(rows, cols) if ok[a, b]]


def _repair_split_merge(P, R, pairs, slack_frac: float = 0.5, min_explained: float = 0.5):
    """ Join unpaired lines to paired groups where the model split / merged lines differently.

    Returns (edits of the paired groups, unpaired pred line idx, unpaired ref line idx).
    """
    gp = [P[i] for i, _, _ in pairs]
    gr = [R[j] for _, j, _ in pairs]
    gcost = [d for _, _, d in pairs]
    gver = [0] * len(pairs)
    paired_p = {i for i, _, _ in pairs}
    paired_r = {j for _, j, _ in pairs}
    # empty / whitespace lines never join a group
    cand = [('r', j) for j in range(len(R)) if j not in paired_r and R[j].strip()]
    cand += [('p', i) for i in range(len(P)) if i not in paired_p and P[i].strip()]
    memo = {}

    def score(side, u, g):
        """ Best (gain, joined text, edits) of joining line u of `side` to group g, None if not worth it. """
        s = R[u] if side == 'r' else P[u]
        key = (side, s, g, gver[g])
        if key not in memo:
            this, that = (gr, gp) if side == 'r' else (gp, gr)
            best = None
            if len(that[g]) - len(this[g]) >= slack_frac * len(s):
                base = len(s) + 1  # cost of leaving the line unpaired
                limit = min(gcost[g] - min_explained * len(s), gcost[g] + base - 1)
                for new in (this[g] + '\n' + s, s + '\n' + this[g]):
                    if abs(len(that[g]) - len(new)) > limit:
                        continue
                    d = _RfLevenshtein.distance(that[g], new, score_cutoff=int(limit) + 1)
                    gain = gcost[g] + base - d
                    if d <= limit and gain > 0 and (best is None or gain > best[0]):
                        best = (gain, new, d)
            memo[key] = best
        return memo[key]

    gains = {k: {g: r for g in range(len(gp)) if (r := score(*k, g)) is not None} for k in cand}
    while True:
        best = None
        for k, per_group in gains.items():
            for g, r in per_group.items():
                if best is None or r[0] > best[0][0] or (r[0] == best[0][0] and (k, g) < best[1:]):
                    best = (r, k, g)
        if best is None:
            break
        (_, new, d), k, g = best
        if k[0] == 'r':
            gr[g] = new
        else:
            gp[g] = new
        gcost[g], gver[g] = d, gver[g] + 1
        del gains[k]
        for k2, per_group in gains.items():  # group g changed, rescore it
            r = score(*k2, g)
            if r is None:
                per_group.pop(g, None)
            else:
                per_group[g] = r
    joined_r = {u for s, u in cand if s == 'r'} - {u for s, u in gains if s == 'r'}
    joined_p = {u for s, u in cand if s == 'p'} - {u for s, u in gains if s == 'p'}
    un_p = [i for i in range(len(P)) if i not in paired_p and i not in joined_p]
    un_r = [j for j in range(len(R)) if j not in paired_r and j not in joined_r]
    return sum(gcost), un_p, un_r


def line_matched_edits(
        prediction: str,
        reference: str,
        gate: float = 0.2,
        min_explained: float = 0.5,
) -> Tuple[int, List[Tuple[int, int, int]]]:
    """ Order robust edits of one page, returns (edits, anchor pairs [(pred line, ref line, lev)]). """
    P, R = prediction.split('\n'), reference.split('\n')
    # 1. identical lines
    ref_pos = {}
    for j, r in enumerate(R):
        ref_pos.setdefault(r, []).append(j)
    pairs, rest_p = [], []
    for i, p in enumerate(P):
        if ref_pos.get(p):
            pairs.append((i, ref_pos[p].pop(0), 0))
        else:
            rest_p.append(i)
    used = {j for _, j, _ in pairs}
    rest_r = [j for j in range(len(R)) if j not in used]
    # 2. gated optimal assignment
    pairs += [
        (rest_p[a], rest_r[b], d)
        for a, b, d in _assign_lines([P[i] for i in rest_p], [R[j] for j in rest_r], gate)]
    # 3. split / merge repair
    edits, un_p, un_r = _repair_split_merge(P, R, pairs, min_explained=min_explained)
    # 4. leftovers: substitutions, then insertions / deletions
    subs = _assign_lines([P[i] for i in un_p], [R[j] for j in un_r])
    edits += sum(d for _, _, d in subs)
    sub_p, sub_r = {a for a, _, _ in subs}, {b for _, b, _ in subs}
    edits += sum(len(P[i]) + 1 for k, i in enumerate(un_p) if k not in sub_p)
    edits += sum(len(R[j]) + 1 for k, j in enumerate(un_r) if k not in sub_r)
    return edits, pairs


def line_order_tau(pairs, reference: str, max_line_err: float = 0.3, min_len: int = 3) -> float:
    """ Kendall tau between predicted and reference line order over anchor pairs (stage 1 / 2, ref line len >=
    min_len, line CER <= max_line_err), nan w/ < 2 anchors. """
    R = reference.split('\n')
    a = sorted((i, j) for i, j, d in pairs if len(R[j].strip()) >= min_len and d <= max_line_err * len(R[j]))
    if len(a) < 2:
        return float('nan')
    return _kendall_tau(np.arange(len(a)), [j for _, j in a])


def line_matched_error_rates(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """ Corpus lm_cer (line matched CER), lm_cer_gap (CER - lm_cer), and mean page line order tau (over `order_pages` pages w/ >= 2
    anchor lines). """
    assert _RfLevenshtein is not None, 'line matched metrics require rapidfuzz'
    edits = cer_edits = ref_chars = 0
    taus = []
    for p, r in zip(predictions, references):
        e_cer = _RfLevenshtein.distance(p, r)
        e_lm, pairs = line_matched_edits(p, r)
        edits += min(e_cer, e_lm)
        cer_edits += e_cer
        ref_chars += len(r)
        taus.append(line_order_tau(pairs, r))
    taus = np.array(taus, dtype=np.float64)
    valid = np.isfinite(taus)
    n = max(ref_chars, 1)
    return {
        'lm_cer': edits / n,
        'lm_cer_gap': (cer_edits - edits) / n,
        'order_tau': float(taus[valid].mean()) if valid.any() else float('nan'),
        'order_pages': int(valid.sum()),
    }
