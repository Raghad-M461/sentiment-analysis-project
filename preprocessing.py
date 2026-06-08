"""
Reusable text-preprocessing pipeline for sentiment analysis.

Every transformation is an independently toggleable flag. This is deliberate:
the point of the task is not to find one "best" pipeline but to be able to run
controlled experiments — turn a single step on/off, hold everything else fixed,
and measure how that one decision moves classification performance.

Design choices that matter for *sentiment* specifically:
  * Negation cues ("not", "never", "n't", ...) are preserved and used to mark
    the words inside their scope (the classic Pang, Lee & Vaithyanathan 2002
    approach), so "not good" and "good" don't collapse into the same features.
  * Stop-word removal uses a sentiment-safe list: generic stop lists delete
    "not", "no", "but", "very", etc., which carry or modify polarity, so we
    explicitly keep those.
  * Lemmatization is preferred over stemming by default (real words, less
    aggressive conflation), but both are available for comparison.

Usage:
    from preprocessing import preprocess
    preprocess("I did NOT like this movie at all!!!")
    preprocess(text, remove_stopwords=False, lemmatize=False, stem=True)
    preprocess(text, return_tokens=True)
"""

from __future__ import annotations

import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


# --------------------------------------------------------------------------- #
# One-time resource bootstrap. Safe to call repeatedly; only downloads if the
# resource is missing, so importing the module never re-fetches data.
# --------------------------------------------------------------------------- #
def _ensure_nltk_data() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_data()

# Reuse single instances — constructing these per call is wasteful.
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()

# Prefix attached to tokens that fall inside a negation scope.
_NEG_PREFIX = "neg_"

# Negation cues that flip the polarity of words that follow them.
# "n't" is included because word_tokenize splits "don't" -> ["do", "n't"].
NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither",
    "nor", "nowhere", "cannot", "cant", "couldnt", "wouldnt", "shouldnt",
    "wont", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "hadnt", "aint", "n't",
}

# A negation scope ends at clause boundaries / contrast words / punctuation.
_NEGATION_STOPPERS = {".", ",", ";", ":", "!", "?", "but", "however", "though"}

# Words a sentiment model should keep even if a generic stop list removes them:
# negators plus intensifiers and contrast markers all change meaning/polarity.
SENTIMENT_KEEP = NEGATION_WORDS | {
    "very", "too", "so", "more", "most", "only", "just",
    "but", "however", "against", "off", "down", "up",
}


@lru_cache(maxsize=1)
def _sentiment_stopwords() -> frozenset[str]:
    """English stop words minus the tokens that carry sentiment signal."""
    return frozenset(set(stopwords.words("english")) - SENTIMENT_KEEP)


def _mark_negation(tokens: list[str]) -> list[str]:
    """Prefix every token inside a negation scope with ``neg_``.

    A scope opens at a negation cue and closes at the next stopper token,
    so "not very good , but ok" -> not neg_very neg_good , but ok.
    """
    out, negating = [], False
    for tok in tokens:
        if tok in _NEGATION_STOPPERS:
            negating = False
            out.append(tok)
        elif tok in NEGATION_WORDS:
            negating = True
            out.append(tok)  # keep the cue itself as a feature
        else:
            out.append(_NEG_PREFIX + tok if negating else tok)
    return out


def _normalize(token: str, *, lemmatize: bool, stem: bool) -> str:
    """Lemmatize or stem a token while preserving any ``neg_`` prefix."""
    prefix, core = "", token
    if token.startswith(_NEG_PREFIX):
        prefix, core = _NEG_PREFIX, token[len(_NEG_PREFIX):]
    if lemmatize:
        core = _LEMMATIZER.lemmatize(core)
    elif stem:
        core = _STEMMER.stem(core)
    return prefix + core


def preprocess(
    text: str,
    *,
    lowercase: bool = True,
    handle_negation: bool = True,
    remove_punctuation: bool = True,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    stem: bool = False,
    return_tokens: bool = False,
) -> str | list[str]:
    """Clean a single text string for sentiment classification.

    The steps run in a fixed, deliberate order. Negation marking happens
    *before* punctuation and stop-word removal because it relies on both
    negation words and punctuation as scope boundaries.

    Args:
        text:               raw input string.
        lowercase:          fold to lower case.
        handle_negation:    mark tokens inside a negation scope with ``neg_``.
        remove_punctuation: drop tokens made up entirely of punctuation.
        remove_stopwords:   drop stop words (sentiment-safe list).
        lemmatize:          reduce words to their dictionary lemma.
        stem:               reduce words to their Porter stem.
        return_tokens:      return a token list instead of a joined string.

    Returns:
        A cleaned string, or a list of tokens if ``return_tokens`` is True.

    Raises:
        TypeError:  if ``text`` is not a string.
        ValueError: if both ``lemmatize`` and ``stem`` are enabled.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    if lemmatize and stem:
        raise ValueError("enable only one of lemmatize / stem, not both")

    # 1. Lowercase
    if lowercase:
        text = text.lower()

    # 2. Tokenize
    tokens = word_tokenize(text)

    # 3. Negation marking (before steps that consume punctuation/negators)
    if handle_negation:
        tokens = _mark_negation(tokens)

    # 4. Drop pure-punctuation tokens
    if remove_punctuation:
        tokens = [t for t in tokens if not all(ch in string.punctuation for ch in t)]

    # 5. Remove stop words (compare against the un-prefixed core word)
    if remove_stopwords:
        stop = _sentiment_stopwords()
        tokens = [
            t for t in tokens
            if (t[len(_NEG_PREFIX):] if t.startswith(_NEG_PREFIX) else t) not in stop
        ]

    # 6. Normalize word forms
    if lemmatize or stem:
        tokens = [_normalize(t, lemmatize=lemmatize, stem=stem) for t in tokens]

    return tokens if return_tokens else " ".join(tokens)


if __name__ == "__main__":
    samples = [
        "I did NOT like this movie at all!!!",
        "The food was absolutely amazing and the staff were very friendly.",
        "It's not bad, but I wouldn't watch it again.",
    ]
    for s in samples:
        print(f"\nRAW : {s}")
        print(f"OUT : {preprocess(s)}")
