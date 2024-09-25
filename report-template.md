
---

# Module 12 Report Template

## Overview of the Analysis

The purpose of this analysis was to build a machine learning model that predicts the creditworthiness of borrowers based on historical lending data. The dataset consisted of features such as loan size, interest rate, borrower income, and other financial metrics, with the target variable being `loan_status` (0 for healthy loans and 1 for high-risk loans). 

### Financial Information:
- **Target Variable (`loan_status`)**: Indicates whether a loan is healthy (0) or high-risk (1).
- **Features (`X`)**: Includes variables like `loan_size`, `interest_rate`, `borrower_income`, `debt_to_income`, `num_of_accounts`, and `total_debt`.

The analysis followed several stages:
1. **Data Preprocessing**: The data was split into training and testing sets (80% training, 20% testing).
2. **Model Building**: A logistic regression model was trained using the training set (`X_train`, `y_train`).
3. **Model Evaluation**: The model's performance was evaluated using a confusion matrix and classification report.

The goal of the analysis was to determine the model's ability to predict whether a loan was healthy or high-risk, based on the available financial data.

## Results

### Logistic Regression Model:
- **Confusion Matrix**:
  ```
  [[14924    77]
   [   31   476]]
  ```
  
- **Classification Report**:
  ```
                precision    recall  f1-score   support

           0       1.00      0.99      1.00     15001
           1       0.86      0.94      0.90       507

    accuracy                           0.99     15508
   macro avg       0.93      0.97      0.95     15508
weighted avg       0.99      0.99      0.99     15508
  ```

- **Accuracy**: 99%
- **Precision for Class 0 (Healthy Loans)**: 100%
- **Recall for Class 0 (Healthy Loans)**: 99%
- **Precision for Class 1 (High-risk Loans)**: 86%
- **Recall for Class 1 (High-risk Loans)**: 94%

## Summary

The logistic regression model performed exceptionally well, achieving an overall accuracy of 99%. It was particularly effective at predicting healthy loans, with a precision of 100% and a recall of 99%. The model also did well in predicting high-risk loans, with a precision of 86% and a recall of 94%. This means that the model was able to correctly identify most of the high-risk loans, although there were some false positives.

### Recommendation:
The logistic regression model is well-suited for this task due to its high accuracy and strong performance in both precision and recall for high-risk loans. Given that false negatives (failing to identify a high-risk loan) are critical for a lending service, this model is recommended for use as it correctly identifies the majority of high-risk loans. However, there is room for improvement, especially in minimizing false positives.

---
