"""
Part 2 - N-Gram Enhancement & Evaluation

This file was created to complete Part 2 of the assignment.

The task requires evaluating three n-gram configurations:

1. Unigrams (1,1)
2. Unigrams + Bigrams (1,2)
3. Unigrams + Bigrams + Trigrams (1,3)

Each configuration is tested using 5-fold cross-validation, which means the dataset is divided into 5 parts. The model is trained and tested
5 times using different data splits, and the average accuracy is reported.

The file also includes negation test cases such as "not good" and "not bad" to evaluate how well each configuration handles sentiment.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

# This is the dataset I used for the sentiment analysis experiment it contains 80 feedback sentences: 40 positive and 40 negative

texts = [
    'A fantastic quality overall.', 'A friendly update overall.', 'A great website overall.',
    'A helpful delivery overall.', 'A perfect service overall.', 'A reliable service overall.',
    'A wonderful onboarding overall.', 'An impressive product overall.', 'Honestly the dashboard is great.',
    'Honestly the service is impressive.', 'Honestly the setup process is perfect.', 'I found the delivery really impressive.',
    'I found the experience really outstanding.', 'I found the platform really great.', 'I found the service really wonderful.',
    'I found the tool really impressive.', 'I found the update really reliable.', 'Such an impressive platform.',
    'Such an impressive support team.', 'Such a reliable service.', 'The dashboard felt reliable to me.',
    'The experience felt helpful to me.', 'The experience was friendly.', 'The interface felt great to me.',
    'The interface was perfect.', 'The price felt perfect to me.', 'The price was excellent.',
    'The quality felt reliable to me.', 'The service felt outstanding to me.', 'The service was friendly.',
    'The software was excellent.', 'The update felt wonderful to me.', 'Their dashboard was reliable.',
    'Their delivery was smooth.', 'Their experience was outstanding.', 'Their onboarding was friendly.',
    'Their team was smooth.', 'Their tool was impressive.', 'Their update was helpful.', 'Their update was reliable.',
    'A broken onboarding overall.', 'A confusing platform overall.', 'A disappointing product overall.',
    'A frustrating dashboard overall.', 'A frustrating delivery overall.', 'A frustrating website overall.',
    'A poor quality overall.', 'A poor website overall.', 'A rude tool overall.', 'A slow delivery overall.',
    'A slow experience overall.', 'An unreliable app overall.', 'Honestly the customer service is rude.',
    'Honestly the dashboard is unreliable.', 'Honestly the software is frustrating.', 'Honestly the tool is unreliable.',
    'I found the customer service really frustrating.', 'I found the dashboard really rude.', 'I found the delivery really terrible.',
    'I found the onboarding really horrible.', 'I found the price really broken.', 'I found the price really horrible.',
    'I found the update really broken.', 'Such a disappointing software.', 'Such a frustrating dashboard.',
    'Such a rude dashboard.', 'Such a rude software.', 'Such a slow quality.', 'Such a slow website.',
    'Such a terrible product.', 'Such an unreliable tool.', 'The interface felt disappointing to me.',
    'The platform was frustrating.', 'The service felt useless to me.', 'The support team felt terrible to me.',
    'The tool felt frustrating to me.', 'The website was horrible.', 'Their customer service was broken.',
    'Their experience was poor.', 'Their product was awful.',
]
labels = ["Positive"] * 40 + ["Negative"] * 40

# this is the 3 configurations it will compare

CONFIGS = {
    "Config 1: Unigrams (1,1)":            (1, 1),
    "Config 2: Unigrams + Bigrams (1,2)":  (1, 2),
    "Config 3: Uni + Bi + Trigrams (1,3)": (1, 3),
}

# here the dataset is split into 5 parts and tested 5 times ( 5 fold cross-validation )

splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def build_model(ngram_range):
    return make_pipeline(
        TfidfVectorizer(ngram_range=ngram_range),
        LogisticRegression(max_iter=1000),
    )


def comparative_evaluation():
    print("=" * 60)
    print("PART B - COMPARATIVE EVALUATION (5-fold CV accuracy)")
    print("=" * 60)
    for name, ngram_range in CONFIGS.items():
        scores = cross_val_score(build_model(ngram_range), texts, labels, cv=splitter)
        folds = [f"{s * 100:.0f}%" for s in scores]
        print(f"{name:36s} mean = {scores.mean() * 100:5.1f}%   folds = {folds}")
    print()


def negation_testing():
    print("=" * 60)
    print("PART C - NEGATION STRESS TEST")
    print("=" * 60)
    cases = [
        ("The product is not good",        "Negative"),
        ("The service was not bad at all",  "Positive"),
        ("Not happy with the support",      "Negative"),
        ("Not the worst experience",        "Positive"),
    ]
    for name, ngram_range in CONFIGS.items():
        model = build_model(ngram_range).fit(texts, labels)
        print(f"\n{name}")
        for sentence, expected in cases:
            pred = model.predict([sentence])[0]
            flag = "PASS" if pred == expected else "FAIL"
            print(f"  [{flag}] expected={expected:8s} predicted={pred:8s} | {sentence}")
    print()


def vocabulary_check():
  
    """This checks whether the words used in the negation tests exist in the dataset."""
  
    print("=" * 60)
    print("VOCABULARY COVERAGE CHECK (why the negation test is misleading)")
    print("=" * 60)
    vocab = set(TfidfVectorizer(ngram_range=(1, 3)).fit(texts).get_feature_names_out())
    for word in ["not", "good", "bad", "happy", "worst", "not good", "not bad"]:
        status = "IN vocabulary" if word in vocab else "NOT in vocabulary (model is blind to it)"
        print(f"  '{word}': {status}")
    print()


if __name__ == "__main__":
    comparative_evaluation()
    negation_testing()
    vocabulary_check()
