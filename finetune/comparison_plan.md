# Fair Comparison Plan

## Frozen Test Set

Every model must be evaluated on the same frozen test set. The test examples and
labels must remain unchanged after the split is created. Using one fixed test set
ensures that differences in results come from the models rather than from easier
or harder evaluation samples.

The frozen test set must not be used for training, feature selection,
hyperparameter tuning, early stopping, or checkpoint selection. It should be
used only once for the final comparison.

## Data Leakage

Data leakage happens when information from the validation or test set influences
model training. Leakage can produce unrealistically high scores that do not
represent performance on unseen data. Examples include training on duplicated
test examples, selecting a checkpoint using test accuracy, fitting preprocessing
steps on all data before splitting, or repeatedly changing the model after
checking test results.

Training decisions will therefore use only the training and validation sets. The
frozen test set will remain separate until the final evaluation.

## Comparison Metrics

The following metrics will be reported for every compared model:

- **Accuracy:** the proportion of all test examples predicted correctly.
- **Macro-F1:** the average F1-score across classes, giving each class equal
  importance.
- **Per-class recall:** the proportion of actual examples from each sentiment
  class correctly identified.
- **Model size:** the storage size of the saved model files.
- **Average inference latency:** the average time required for one prediction
  under the same evaluation environment.

All models must use the same frozen test examples, label order, preprocessing
rules, and latency-measurement procedure.
