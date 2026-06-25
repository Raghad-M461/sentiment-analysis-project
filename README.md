## Run a Prediction

First, train and save the model:

```bash
python train_and_save.py
```

Then run a prediction:

```bash
python predict.py "Your sentence here"
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
