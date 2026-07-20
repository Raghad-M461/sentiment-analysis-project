# DistilBERT Fine-Tuning Notes

## Model Configuration

- **Base model:** distilbert-base-uncased
- **Epochs:** 4
- **Learning rate:** 2e-5
- **Training batch size:** 8
- **Evaluation strategy:** Every epoch
- **Save strategy:** Every epoch
- **Early stopping:** Enabled (patience = 1)
- **Best model selection:** Lowest validation loss

## Training Results

| Epoch | Validation Loss | Validation Accuracy |
|------:|----------------:|--------------------:|
| 1 | 1.0609 | 54.84% |
| 2 | 0.9076 | 83.87% |
| 3 | 0.7357 | 90.32% |
| 4 | 0.6788 | 90.32% |

## Training Analysis

The model was fine-tuned for four epochs using the Hugging Face Trainer API. Validation loss decreased consistently throughout training, indicating that the model continued learning effectively. Validation accuracy improved from 54.84% in the first epoch to 90.32% by the third epoch and remained stable during the fourth epoch. Since the validation loss continued to decrease, there was no evidence of overfitting during this training run.

## Best Checkpoint

The best model checkpoint was automatically selected using the lowest validation loss and saved in the `finetune/best_checkpoint` directory.

## Conclusion

The fine-tuning process completed successfully. The trained DistilBERT model achieved a final validation accuracy of **90.32%** and is ready for evaluation on the frozen test dataset in the next stage of the project.
