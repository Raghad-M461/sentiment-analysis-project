texts = [
    "The service was excellent",
    "Terrible customer support",
    "I loved the experience",
    "Very disappointing experience"
]

labels = [
    "Positive",
    "Negative",
    "Positive",
    "Negative"
]

print("Sentiment Analysis Dataset Loaded Successfully")

for text, label in zip(texts, labels):
    print(text, "->", label)
