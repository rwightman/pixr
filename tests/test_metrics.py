from pixr.utils import anls_score, average_normalized_levenshtein_similarity, text_error_rates


def test_anls():
    assert anls_score('  The Answer ', ['the answer']) == 1.0
    assert anls_score('abc', ['xyz']) == 0.0
    assert abs(anls_score('answe', ['answer']) - (1 - 1 / 6)) < 1e-6
    assert average_normalized_levenshtein_similarity([['a'], ['b', 'c']], ['a', 'c']) == 1.0


def test_error_rates():
    r = text_error_rates(['hello wrld'], ['hello world'])
    assert abs(r['cer'] - 1 / 11) < 1e-6 and r['wer'] == 0.5


def test_linear_sum_assignment_brute_force():
    import itertools
    import numpy as np
    from pixr.utils.metrics import linear_sum_assignment
    rng = np.random.default_rng(0)
    for n, m in [(1, 1), (2, 3), (3, 2), (4, 4), (3, 6), (6, 3)]:
        for t in range(10):
            cost = rng.integers(-5, 3, (n, m)).astype(float) if t % 2 else rng.normal(size=(n, m))
            rows, cols = linear_sum_assignment(cost)
            k = min(n, m)
            assert len(rows) == k and len(set(rows)) == k and len(set(cols)) == k
            if n <= m:
                best = min(sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
            else:
                best = min(sum(cost[p[j], j] for j in range(m)) for p in itertools.permutations(range(n), m))
            assert np.isclose(cost[rows, cols].sum(), best)


def test_line_matched_error_rates():
    import random
    from pixr.utils import line_matched_error_rates
    lines = [f'line {i} of the page w/ some text {i * 7}' for i in range(30)]
    ref = '\n'.join(lines)
    shuffled = lines[:]
    random.Random(0).shuffle(shuffled)
    r = line_matched_error_rates(['\n'.join(shuffled)], [ref])
    cer = text_error_rates(['\n'.join(shuffled)], [ref])['cer']
    assert r['lm_cer'] == 0 and cer > 0.05 and abs(r['lm_cer_gap'] - cer) < 1e-9 and r['order_tau'] < 0.5
    # in order: lm_cer == CER, tau 1
    r = line_matched_error_rates([ref], [ref])
    assert r['lm_cer'] == 0 and r['lm_cer_gap'] == 0 and r['order_tau'] == 1 and r['order_pages'] == 1
    reversed_ = line_matched_error_rates(['\n'.join(lines[::-1])], [ref])
    assert reversed_['lm_cer'] == 0 and reversed_['order_tau'] == -1
    # char noise in order: lm_cer == CER
    noisy = '\n'.join(l.replace('of', 'ot') for l in lines)
    r, t = line_matched_error_rates([noisy], [ref]), text_error_rates([noisy], [ref])
    assert abs(r['lm_cer'] - t['cer']) < 1e-9 and r['lm_cer_gap'] == 0
    # repetition loop / omission charged in full (lm_cer == CER)
    loop = '\n'.join(lines[:10] + [lines[10]] * 40)
    r, t = line_matched_error_rates([loop], [ref]), text_error_rates([loop], [ref])
    assert r['lm_cer'] > 0.5 and abs(r['lm_cer'] - t['cer']) < 0.02
    # a split line costs ~1 edit
    split = '\n'.join(lines[:5] + ['line 5 of the page', 'w/ some text 35'] + lines[6:])
    r = line_matched_error_rates([split], [ref])
    assert r['lm_cer'] * len(ref) <= 2
    # lm_cer <= CER per page, empty pred / ref
    r = line_matched_error_rates(['', 'abc', '\n'.join(shuffled[:12])], [ref, '', ref])
    t = text_error_rates(['', 'abc', '\n'.join(shuffled[:12])], [ref, '', ref])
    assert r['lm_cer'] <= t['cer'] + 1e-9


def test_summary_csv_new_columns(tmp_path):
    import csv
    from pixr.framework.monitor import SummaryCsv
    s = SummaryCsv(str(tmp_path))
    s.update({'interval': 0, 'eval_cer': 0.5})
    s.update({'interval': 1, 'eval_cer': 0.4})
    s = SummaryCsv(str(tmp_path))  # resume w/ new metrics, missing ones left empty
    s.update({'interval': 2, 'eval_cer': 0.3, 'eval_lm_cer': 0.2})
    s.update({'interval': 3, 'eval_lm_cer': 0.1, 'eval_cer': 0.25})
    with open(tmp_path / 'summary.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    assert [r['interval'] for r in rows] == ['0', '1', '2', '3']
    assert [r['eval_lm_cer'] for r in rows] == ['', '', '0.2', '0.1'] and rows[3]['eval_cer'] == '0.25'
