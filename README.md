# Hand Gesture Recognition

Hand gesture classification project built from 21 hand landmark coordinates.

## Project Summary

This project classifies hand gestures using landmark points extracted from hand images. The workflow covers:

- data loading
- landmark visualization
- preprocessing and normalization
- train/test split
- model training and comparison
- final model selection

## Preprocessing Choice

The detected hands appear at different positions and scales in the image. To reduce that variation, the notebook applies two preprocessing steps:

1. Geometric normalization
   - recenter each sample so the wrist is the origin
   - scale landmarks using the distance from the wrist to the middle-finger tip
   - keep `z` unchanged because it is already normalized by MediaPipe

2. Feature normalization
   - apply train-set z-score normalization before fitting the models
   - this is useful for SVM and KNN because they are sensitive to feature scale

This makes the training and test samples more consistent while preserving hand shape.

## Model Choice

The selected model is Random Forest.

Reason:

- it achieved the highest test accuracy
- it achieved the highest macro F1 score
- it gave the strongest overall classification performance on this dataset

KNN is faster to tune, but Random Forest produced the best final results.

## Model Comparison

| Model | Accuracy | Macro F1 | Search Time (s) |
| --- | ---: | ---: | ---: |
| Random Forest | 0.9811 | 0.9809 | 222.62 |
| SVM (RBF) | 0.9723 | 0.9721 | 59.71 |
| KNN | 0.9470 | 0.9476 | 6.96 |

## Files

- `hand_landmarks.ipynb`: full notebook for preprocessing, training, and evaluation
- `hand_landmarks_data.csv`: dataset of hand landmarks and labels

## Run

1. Open `hand_landmarks.ipynb`.
2. Run the cells in order.
3. Review the model comparison output and final selected model.