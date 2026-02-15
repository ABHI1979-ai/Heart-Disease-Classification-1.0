# Heart Disease Classification Project

## a. Problem Statement
Predicting the presence of heart disease using clinical features to assist in early medical intervention.

## b. Dataset Description
- **Size**: 920 instances, 13 features.
- **Target**: `num` (0 = No Disease, 1 = Disease).

## c. Model Comparison Table
| ML Model Name       |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.7826 | 0.8606 |      0.8632 |   0.7523 | 0.8039 | 0.5693 |
| Decision Tree       |     0.8043 | 0.8016 |      0.8476 |   0.8165 | 0.8318 | 0.5988 |
| kNN                 |     0.7989 | 0.8596 |      0.8673 |   0.7798 | 0.8213 | 0.5973 |
| Naive Bayes         |     0.7826 | 0.8749 |      0.871  |   0.7431 | 0.802  | 0.5731 |
| Random Forest       |     0.837  | 0.9054 |      0.8762 |   0.844  | 0.8598 | 0.6658 |
| XGBoost             |     0.8587 | 0.8986 |      0.9192 |   0.8349 | 0.875  | 0.7177 |

## d. Observations
- **XGBoost** provided the highest Accuracy and MCC, making it the most reliable for binary classification here.
- **Random Forest** had the highest AUC, showing excellent separation between classes.
