"""
Reusable preprocessing function created for Task 6.

This function applies common NLP preprocessing techniques such as lowercasing, tokenization, stop-word removal, stemming, lemmatization, punctuation removal, and negation handling.

The settings can be changed easily so I can test how different preprocessing methods affect sentiment analysis results.
"""

from __future__ import annotations

import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


# it will download NLTK resources needed for preprocessing and resources that already exist will not be downloaded again.

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

_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()

_NEG_PREFIX = "neg_"

# Negation words such as not, no, never, and n't. NLTK separates contractions like "don't" into "do" and "n't".

NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither",
    "nor", "nowhere", "cannot", "cant", "couldnt", "wouldnt", "shouldnt",
    "wont", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "hadnt", "aint", "n't",
}

_NEGATION_STOPPERS = {".", ",", ";", ":", "!", "?", "but", "however", "though"}


SENTIMENT_KEEP = NEGATION_WORDS | {
    "very", "too", "so", "more", "most", "only", "just",
    "but", "however", "against", "off", "down", "up",
}


@lru_cache(maxsize=1)
def _sentiment_stopwords() -> frozenset[str]:
    return frozenset(set(stopwords.words("english")) - SENTIMENT_KEEP)


def _mark_negation(tokens: list[str]) -> list[str]:
   
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
   token
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
   
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    if lemmatize and stem:
        raise ValueError("enable only one of lemmatize / stem, not both")

    # 1. Lowercase
    if lowercase:
        text = text.lower()

    # 2. Tokenize
    tokens = word_tokenize(text)

    # 3. Negation marking 
    if handle_negation:
        tokens = _mark_negation(tokens)

    # 4. Remove punctuation-only tokens
    if remove_punctuation:
        tokens = [t for t in tokens if not all(ch in string.punctuation for ch in t)]

    # 5. Checks the original word before removing stop words
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
