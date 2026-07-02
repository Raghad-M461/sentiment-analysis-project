
import pytest
from predict import predict_sentiment

REGRESSION_CASES = [
    (
        "The product quality is outstanding and the delivery was fast",
        "Positive",
        "clear positive sentence",
    ),
    (
        "This is not good at all, very disappointed with the service",
        "Negative",
        "negation case — 'not good' should be Negative",
    ),
    (
        "The app works fine but the customer support could be better",
        "Negative",
        "mixed sentiment — model picks the negative side",
    ),
    (
        "Absolutely terrible experience, would not recommend to anyone",
        "Negative",
        "clear negative sentence",
    ),
    (
        "Not bad, actually quite impressed with how well it performs",
        "Positive",
        "negation of negative — 'not bad' should be Positive",
    ),
]


@pytest.mark.parametrize("text, expected_label, note", REGRESSION_CASES)
def test_label(text: str, expected_label: str, note: str):
   
    result = predict_sentiment(text)

    assert "label" in result, "response must contain 'label'"
    assert "confidence" in result, "response must contain 'confidence'"
    assert isinstance(result["confidence"], float), "confidence must be a float"
    assert 0.0 <= result["confidence"] <= 1.0, "confidence must be between 0 and 1"
    assert result["label"] == expected_label, (
        f"REGRESSION on: '{text}'\n"
        f"  note: {note}\n"
        f"  expected: {expected_label}\n"
        f"  got:      {result['label']} (confidence: {result['confidence']:.2%})"
    )


def test_empty_string_raises():
    with pytest.raises(ValueError):
        predict_sentiment("")


def test_whitespace_raises():
    with pytest.raises(ValueError):
        predict_sentiment("   ")


def test_confidence_range():
    result = predict_sentiment("the service was excellent")
    assert 0.0 <= result["confidence"] <= 1.0


def test_label_is_valid_class():
    valid_labels = {"Positive", "Negative", "Neutral"}
    result = predict_sentiment("the product arrived on time")
    assert result["label"] in valid_labels, (
        f"unexpected label: {result['label']}"
    )
