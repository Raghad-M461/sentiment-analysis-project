# Inference — Notes and Model Choice

**Scripts:** `train_and_save.py`, `predict.py`, `test_predictions.py`

---

# Part 1 — Concepts

## Training code vs. inference code

Training code and inference code have different purposes and should be kept separate.

Training code loads the dataset, converts the text into features, trains the model, evaluates its performance, and saves the trained model. This process is only done when creating or updating the model, so it can take longer.

Inference code is used after the model has already been trained. It loads the saved model, accepts new input text, and returns a prediction. It should not retrain the model or access the training dataset. Keeping inference separate makes the prediction function reusable, faster, and easier to include in other applications.

---

## Why save the trained model?

Training the model every time a prediction is needed would waste time because the model does not change between predictions.

Instead, the trained Logistic Regression model is saved using `joblib`. When a prediction is requested, `predict.py` loads the saved model and reuses it. This makes predictions much faster and avoids unnecessary retraining.

The sentence embedding model is not saved because it is already a pretrained model that can be loaded directly from the Sentence Transformers library. Only the classifier trained on this dataset needs to be stored.

---

## What should a prediction function return?

A good prediction interface should return both:

* the predicted sentiment label
* the confidence score

The label tells the user whether the text is Positive, Negative, or Neutral. The confidence score shows how certain the model is about that prediction.

Showing the confidence score is useful because low-confidence predictions can be reviewed manually or treated with more caution in real applications.

The prediction function returns:

```python
{"label": str, "confidence": float}
```

where the confidence value comes from Logistic Regression's `predict_proba()` method.

---

# Part 2 — Model Choice

## Chosen model

**Sentence Embeddings (all-MiniLM-L6-v2) + Logistic Regression**

The model comparison used the same dataset, classifier, and cross-validation split for every feature representation.

| Features                     |  Accuracy |        F1 |
| ---------------------------- | --------: | --------: |
| TF-IDF                       |     69.5% |     68.2% |
| Averaged GloVe               |     74.3% |     73.8% |
| Sentence Embeddings (MiniLM) | **82.4%** | **82.1%** |

Sentence embeddings achieved the highest Accuracy and F1 score, making it the best-performing model.

The error analysis also showed that it handled negation better than TF-IDF. For example, phrases such as **"not good"** and **"not bad"** were interpreted more accurately because the model considers the meaning of the whole sentence instead of relying mainly on individual words.

For these reasons, the Sentence Embedding model was selected for the final prediction system.

---

# Part 2 — Test Predictions

The model was trained on the full 210-sample dataset using `train_and_save.py` and then tested using `test_predictions.py`.

| # | Sentence                                                      | Label    | Confidence |
| - | ------------------------------------------------------------- | -------- | ---------- |
| 1 | The product quality is outstanding and the delivery was fast  | Positive | 78%        |
| 2 | This is not good at all, very disappointed with the service   | Negative | 57%        |
| 3 | The app works fine but the customer support could be better   | Negative | 54%        |
| 4 | Absolutely terrible experience, would not recommend to anyone | Negative | 53%        |
| 5 | Not bad, actually quite impressed with how well it performs   | Positive | 66%        |

## Discussion

The model correctly predicted both negation examples:

* **"not good at all"** was classified as **Negative**.
* **"not bad"** was classified as **Positive**.

The mixed-sentiment sentence was classified as **Negative** with a confidence of **54%**. This sentence contains both positive and negative opinions, making it more difficult to classify. The lower confidence score reflects that uncertainty.

Overall, the confidence scores ranged from **53% to 78%**. This is reasonable for a three-class sentiment classification problem trained on a relatively small dataset.

---

# How to Run a Prediction

Install the required packages:

```bash
pip install -r requirements.txt
```

Train and save the model:

```bash
python train_and_save.py
```

Run a prediction:

```bash
python predict.py "your sentence here"
```

Example:

```bash
python predict.py "Not bad, actually quite impressed with how well it performs"
```

Example output:

```text
Input      : Not bad, actually quite impressed with how well it performs
Label      : Positive
Confidence : 66.00%
```
