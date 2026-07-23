# Fine-Tuning Limitations

The final comparison demonstrated that the fine-tuned DistilBERT model achieved the best overall performance on the prepared sentiment analysis dataset. However, these results should be interpreted within the limitations of this project. The evaluation provides evidence that fine-tuning improved sentiment classification performance under the conditions of this experiment, but it does not guarantee the same results in every real-world scenario.

## What the project demonstrates

- Fine-tuning DistilBERT improved sentiment classification performance compared with the TF-IDF and MiniLM baseline models when evaluated on the same frozen test set.
- Using a frozen test set enabled a fair, reproducible, and unbiased comparison between all three models.
- The fine-tuned model showed stronger contextual understanding and handled negated sentiment expressions more consistently than the baseline approaches.
- Multiple evaluation metrics, including Accuracy, Macro-F1 score, per-class recall, inference latency, and model size, supported an evidence-based deployment recommendation.

## What the project does not prove

- The dataset is relatively small and may not represent the full diversity of real-world sentiment across different domains.
- The evaluation is based on a single training run. Different random seeds or repeated experiments may produce slightly different results.
- The primary training and evaluation were performed using an English-language sentiment dataset. Although a small number of Arabic examples were explored during testing, multilingual performance was not systematically evaluated and therefore cannot be considered a validated capability.
- Performance on out-of-domain inputs, domain-specific terminology, slang, emojis, mixed-language text, or previously unseen topics has not been comprehensively evaluated.
- The project does not demonstrate long-term production reliability, continuous monitoring, or large-scale deployment performance.

Overall, this project provides strong evidence that fine-tuning DistilBERT is the most effective approach for the prepared dataset and evaluation procedure. However, additional experiments using larger datasets, multiple training runs, broader language coverage, and production-scale testing would be required before generalizing these findings to all real-world sentiment analysis applications.
