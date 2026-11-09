from pixr.utils import anls_score, average_normalized_levenshtein_similarity, text_error_rates


def test_anls():
    assert anls_score('  The Answer ', ['the answer']) == 1.0
    assert anls_score('abc', ['xyz']) == 0.0
    assert abs(anls_score('answe', ['answer']) - (1 - 1 / 6)) < 1e-6
    assert average_normalized_levenshtein_similarity([['a'], ['b', 'c']], ['a', 'c']) == 1.0


def test_error_rates():
    r = text_error_rates(['hello wrld'], ['hello world'])
    assert abs(r['cer'] - 1 / 11) < 1e-6 and r['wer'] == 0.5
