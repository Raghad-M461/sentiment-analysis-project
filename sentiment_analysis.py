from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training dataset
texts = [
    "The service was excellent",
    "I loved the experience",
    "Very professional support team",
    "Terrible customer support",
    "Very disappointing experience",
    "The application crashes constantly"
]

labels = [
    "Positive",
    "Positive",
    "Positive",
    "Negative",
    "Negative",
    "Negative"
]

# Convert text into numbers
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

# Train AI model
model = LogisticRegression()

model.fit(X, labels)

# Test predictions
test_texts = [
    "The support team solved my problem quickly",
    "The software is terrible"
]

test_vectors = vectorizer.transform(test_texts)

predictions = model.predict(test_vectors)

# Print predictions
print("Sentiment Predictions:\n")

for text, prediction in zip(test_texts, predictions):
    print(text)
    print("Prediction:", prediction)
    print()
