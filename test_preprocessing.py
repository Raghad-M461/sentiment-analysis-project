

from preprocessing import preprocess


def test_lowercasing():
    assert "HELLO" not in preprocess("HELLO World", return_tokens=True)


def test_negation_scope_marks_following_words():
    out = preprocess("I do not like it", return_tokens=True)
    assert "not" in out
    assert "neg_like" in out


def test_negation_scope_closes_at_punctuation():
    out = preprocess("not bad, good overall", return_tokens=True)
    assert "neg_bad" in out
    assert "neg_good" not in out
    assert "good" in out


def test_sentiment_stopwords_are_preserved():
   
    out = preprocess("this is not very good", handle_negation=False, return_tokens=True)
    assert "not" in out          # negator kept, not stripped as a stop word
    assert "very" in out         # intensifier kept
    assert "is" not in out       # ordinary stop word still removed


def test_punctuation_removed():
    out = preprocess("great!!!", return_tokens=True)
    assert "!" not in out
    assert "great" in out


def test_stem_and_lemmatize_are_mutually_exclusive():
    try:
        preprocess("x", lemmatize=True, stem=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when both are enabled")


def test_non_string_raises():
    try:
        preprocess(123)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for non-string input")


def test_returns_string_by_default():
    assert isinstance(preprocess("hello world"), str)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
