# DistilBERT Fine-Tuning Notes

## Task Overview

The purpose of this task was to fine-tune `distilbert-base-uncased` for sentiment classification using the prepared training and validation datasets.

The model was trained using the Hugging Face `Trainer` API. Validation evaluation and checkpoint saving were performed after every epoch. The best checkpoint was selected automatically using the lowest validation loss.

## Dataset

- Training examples: 147
- Validation examples: 31
- Number of sentiment labels: 3

The three sentiment labels were:

- Negative
- Neutral
- Positive

The frozen test dataset was not used during training or validation.

## Model Configuration

- Base model: `distilbert-base-uncased`
- Number of epochs: 4
- Learning rate: `2e-5`
- Training batch size: 8
- Validation batch size: 8
- Maximum token length: 128
- Weight decay: 0.01
- Evaluation strategy: Every epoch
- Checkpoint save strategy: Every epoch
- Best model metric: Validation loss
- Early stopping patience: 1
- Random seed: 42
- Training device: CPU

## Role of Hugging Face Trainer

The Hugging Face `Trainer` managed the main training loop, including batch processing, forward and backward passes, parameter updates, evaluation, logging, and checkpoint saving.

`TrainingArguments` controlled the main training settings, including the number of epochs, learning rate, batch size, evaluation frequency, checkpoint-saving strategy, and best-checkpoint selection.

The model was evaluated after every epoch. Because `load_best_model_at_end=True` was enabled, the checkpoint with the lowest validation loss was restored at the end of training rather than simply keeping the last model without comparison.

## Per-Epoch Results

| Epoch | Training Loss | Validation Loss | Validation Accuracy |
|------:|--------------:|----------------:|--------------------:|
| 1 | 1.0946 | 1.0609 | 54.84% |
| 2 | 0.9799 | 0.9076 | 83.87% |
| 3 | 0.7883 | 0.7357 | 90.32% |
| 4 | 0.6587 | 0.6788 | 90.32% |

## Training Curve Interpretation

The model learned successfully because both training loss and validation loss decreased throughout the four epochs. Validation accuracy improved significantly from 54.84% in epoch 1 to 90.32% in epoch 3. Accuracy remained stable during epoch 4, but validation loss continued to decrease, indicating that the model became more confident in its predictions. There was no clear evidence of overfitting because validation loss did not increase while training loss decreased.

## Healthy Learning, Overfitting, and Underfitting

Healthy learning occurs when both training and validation loss decrease and validation accuracy improves. This pattern appeared in the current training run.

Overfitting would occur if training loss continued to decrease while validation loss started increasing or validation accuracy became worse. This was not observed during the four epochs.

Underfitting would occur if both training and validation performance remained poor, with high losses and low accuracy. The final validation accuracy of 90.32% indicates that the model was not underfitting at the end of training.

## Best Epoch and Checkpoint

Epoch 4 was selected as the best epoch because it achieved the lowest validation loss:

- Best validation loss: 0.6788
- Validation accuracy: 90.32%
- Best checkpoint: `finetune/training_output/checkpoint-76`

Epochs 3 and 4 achieved the same validation accuracy, but epoch 4 had a lower validation loss. Therefore, epoch 4 was correctly selected as the best checkpoint.

The final best model was saved in:

`finetune/best_checkpoint/`

The saved checkpoint includes:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_args.bin`

## Early Stopping

Early stopping was enabled with a patience value of 1. It would have stopped training if validation loss failed to improve for one evaluation period.

Early stopping was not triggered because validation loss improved after every epoch:

- Epoch 1: 1.0609
- Epoch 2: 0.9076
- Epoch 3: 0.7357
- Epoch 4: 0.6788

Therefore, the model completed all four planned epochs.

## Final Training Summary

- Final average training loss: 0.8804
- Best validation loss: 0.6788
- Final validation accuracy: 90.32%
- Total training runtime: approximately 22.51 seconds
- Best epoch: 4
- Best checkpoint: `checkpoint-76`

## Issues and Notes

Training completed successfully without any errors.

The model was trained on CPU because GPU (CUDA) was not available. Since the dataset was small, training completed successfully within approximately 23 seconds.

No issues affected the fine-tuning process or the generated checkpoint.

## Conclusion

The fine-tuning run completed successfully and met the task requirements. The model showed healthy learning across all four epochs, with steadily decreasing training and validation losses. Validation accuracy reached 90.32%, and epoch 4 was selected as the best epoch because it achieved the lowest validation loss. The best checkpoint was saved to disk and is ready for evaluation using the frozen test dataset.
