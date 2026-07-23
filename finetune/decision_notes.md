# Decision Notes

## Decision Criteria (Written Before Reviewing Results)

Before comparing the models, the following criteria were defined to ensure that the final recommendation would be based on evidence rather than personal preference.

1. **Prediction Quality**
   - Accuracy and Macro-F1 are the primary performance metrics.
   - Per-class recall is also important to ensure that the model performs consistently across all sentiment classes.

2. **Inference Speed**
   - The model should provide predictions quickly enough for a real-time sentiment analysis service.
   - Lower latency is preferred when accuracy differences are small.

3. **Model Size**
   - Smaller models are easier to deploy, require less storage, and consume fewer computing resources.
   - Large models should provide a meaningful improvement in performance to justify their size.

4. **Maintainability**
   - Simpler models are generally easier to understand, retrain, and maintain.
   - More complex models should only be selected if they provide a clear performance advantage.

## What "Good Enough" Means

A model is considered good enough if it:

- Achieves strong overall accuracy and Macro-F1.
- Maintains balanced recall across all sentiment classes.
- Provides acceptable inference latency for deployment.
- Does not introduce unnecessary complexity for only a small improvement.

## Recommendation Strategy

The final recommendation will not automatically choose the largest or most advanced model. Instead, it will consider the balance between prediction quality, computational cost, latency, model size, and deployment complexity.
