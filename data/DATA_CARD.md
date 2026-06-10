# Data Card – Enriched 3-Class Sentiment Dataset

## Project Information

* Project: Sentiment Analysis Project
* Branch: feature/dataset-expansion
* Approach: Option B – Dataset Enrichment
* Classes: Positive, Negative, Neutral

## Dataset Source

The original dataset contained 80 manually labelled customer feedback sentences (40 Positive and 40 Negative).

For this task, I expanded the dataset by adding 130 new manually labelled sentences. These new samples were created to:

* Add a Neutral class
* Increase dataset size
* Improve language variety
* Include more difficult examples such as sarcasm, negation, and mixed sentiment

No external datasets were used.

## Dataset Statistics

| Class    | Samples |
| -------- | ------- |
| Positive | 70      |
| Negative | 70      |
| Neutral  | 70      |
| Total    | 210     |

The dataset is balanced, with the same number of samples in each class.

## Class Distribution

A bar chart was created to show the distribution of Positive, Negative, and Neutral samples. Each class contains 70 examples.

## Preprocessing

The same preprocessing pipeline from the previous task was used:

* Lowercasing
* Tokenization
* Negation handling
* Stop-word removal
* Lemmatization

Using the same preprocessing allows a fair comparison between the original and expanded datasets.

## Limitations

* The dataset is relatively small (210 samples).
* The added samples were written and labelled manually by one person.
* The dataset only contains English text.
* Real-world customer feedback may be more complex than the examples used in this project.

## Conclusion

The dataset was successfully expanded from 80 to 210 samples. A Neutral class was added, and more diverse examples were included to create a more realistic sentiment analysis task.
