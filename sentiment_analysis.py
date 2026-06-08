from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

from preprocessing import preprocess   # <-- use our reusable pipeline


texts = [
    'A fantastic quality overall.',
    'A friendly update overall.',
    'A great website overall.',
    'A helpful delivery overall.',
    'A perfect service overall.',
    'A reliable service overall.',
    'A wonderful onboarding overall.',
    'An impressive product overall.',
    'Honestly the dashboard is great.',
    'Honestly the service is impressive.',
    'Honestly the setup process is perfect.',
    'I found the delivery really impressive.',
    'I found the experience really outstanding.',
    'I found the platform really great.',
    'I found the service really wonderful.',
    'I found the tool really impressive.',
    'I found the update really reliable.',
    'Such an impressive platform.',
    'Such an impressive support team.',
    'Such a reliable service.',
    'The dashboard felt reliable to me.',
    'The experience felt helpful to me.',
    'The experience was friendly.',
    'The interface felt great to me.',
    'The interface was perfect.',
    'The price felt perfect to me.',
    'The price was excellent.',
    'The quality felt reliable to me.',
    'The service felt outstanding to me.',
    'The service was friendly.',
    'The software was excellent.',
    'The update felt wonderful to me.',
    'Their dashboard was reliable.',
    'Their delivery was smooth.',
    'Their experience was outstanding.',
    'Their onboarding was friendly.',
    'Their team was smooth.',
    'Their tool was impressive.',
    'Their update was helpful.',
    'Their update was reliable.',
    'A broken onboarding overall.',
    'A confusing platform overall.',
    'A disappointing product overall.',
    'A frustrating dashboard overall.',
    'A frustrating delivery overall.',
    'A frustrating website overall.',
    'A poor quality overall.',
    'A poor website overall.',
    'A rude tool overall.',
    'A slow delivery overall.',
    'A slow experience overall.',
    'An unreliable app overall.',
    'Honestly the customer service is rude.',
    'Honestly the dashboard is unreliable.',
    'Honestly the software is frustrating.',
    'Honestly the tool is unreliable.',
    'I found the customer service really frustrating.',
    'I found the dashboard really rude.',
    'I found the delivery really terrible.',
    'I found the onboarding really horrible.',
    'I found the price really broken.',
    'I found the price really horrible.',
    'I found the update really broken.',
    'Such a disappointing software.',
    'Such a frustrating dashboard.',
    'Such a rude dashboard.',
    'Such a rude software.',
    'Such a slow quality.',
    'Such a slow website.',
    'Such a terrible product.',
    'Such an unreliable tool.',
    'The interface felt disappointing to me.',
    'The platform was frustrating.',
    'The service felt useless to me.',
    'The support team felt terrible to me.',
    'The tool felt frustrating to me.',
    'The website was horrible.',
    'Their customer service was broken.',
    'Their experience was poor.',
    'Their product was awful.',
]

labels = ["Positive"] * 40 + ["Negative"] * 40

#    This is the line that connects Task 6 to the model.
texts_clean = [preprocess(t) for t in texts]

model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, texts_clean, labels, cv=splitter)
print("Cross-validation accuracy:", round(scores.mean() * 100, 1), "%")
print()

model.fit(texts_clean, labels)

new_texts = [
    "The support team was friendly and very helpful",
    "The app is terrible and keeps crashing",
    "A wonderful and reliable service",
    "The setup process was confusing and slow",
]

print("Sentiment Predictions:")
print()
for text in new_texts:
    # Preprocess each new sentence the SAME way before predicting,
    # but print the original so the output stays readable.
    prediction = model.predict([preprocess(text)])[0]
    print(text)
    print("Prediction:", prediction)
    print()
