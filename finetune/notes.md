# Fine-Tuning Notes

## Frozen Feature Extraction vs. Fine-Tuning

In Week 4, I used DistilBERT as a frozen feature extractor. The model generated text embeddings, and a Logistic Regression classifier was trained using those embeddings. The DistilBERT weights were not changed.

Fine-tuning is different because the pretrained DistilBERT model and a small classification head are trained together on the sentiment dataset. This allows the model to learn patterns that are specific to the dataset.

## Simple Terms

- **Epoch:** One complete pass through the training dataset.
- **Batch Size:** The number of samples processed before updating the model.
- **Learning Rate:** Controls how much the model changes during training. Fine-tuning uses a small learning rate so the pretrained knowledge is not damaged.
- **Overfitting:** When the model learns the training data too well and performs worse on new data.

## Why Keep the Test Set Frozen?

The test set should not be used during training or model tuning. Keeping it unchanged provides a fair and honest evaluation of the final model.

## Reflection 

Fine-tuning may perform better than the frozen embeddings with Logistic Regression because it updates the pretrained model using the sentiment dataset. This allows the model to better understand the words, context, and sentiment found in the training data. The frozen approach only trains the Logistic Regression classifier while the language model remains unchanged.

However, fine-tuning is not always the best choice. It requires more computing resources, takes longer to train, and can overfit if the dataset is small. If the frozen embeddings already give good performance, the improvement from fine-tuning may be small. In those situations, using frozen embeddings is faster and simpler. The best approach depends on the dataset size, available computing power, and the performance requirements of the project.
