
---

# Credit Risk Classification

## Table of Contents
- [Overview](#overview)
- [Background](#background)
- [Project Structure](#project-structure)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
The **Credit Risk Classification** project aims to build a machine learning model that can predict the creditworthiness of borrowers. Using historical lending activity data from a peer-to-peer lending services company, this project focuses on classifying loans as either high-risk or healthy based on key financial metrics. The objective is to provide insights into loan risks and identify potential defaulters.

## Background
Peer-to-peer lending platforms rely on risk assessment models to evaluate borrower creditworthiness. By predicting the risk of default, such models help in better loan allocation and financial decision-making. The dataset used in this project consists of borrower financial records and their corresponding loan status, where a loan can either be categorized as:
- **Healthy loan (0)**: Low risk of default.
- **High-risk loan (1)**: High probability of default.

This challenge involves using a logistic regression model to classify the loans and assess the model’s performance using various metrics.

## Project Structure
The project is structured into several key steps:
1. **Data Preprocessing**: Split the data into training and testing datasets.
2. **Model Training**: Fit a logistic regression model on the training data.
3. **Model Evaluation**: Evaluate the model using performance metrics such as accuracy, precision, recall, and F1-score.
4. **Credit Risk Analysis Report**: Summarize the findings and provide recommendations.

## Features
- **Data Preprocessing**: Splitting the data into labels (`y`) and features (`X`), and further dividing it into training and testing sets.
- **Logistic Regression Model**: Training a logistic regression model on the dataset.
- **Model Evaluation**: Using a confusion matrix and classification report to evaluate model performance.
- **Credit Risk Report**: A comprehensive report summarizing the model’s performance and its potential use in predicting loan risk.

## Dataset
The dataset used in this project is sourced from the **lending_data.csv** file, which contains historical lending activity data. The primary target variable is the `loan_status` column, where:
- **0** indicates a healthy loan.
- **1** indicates a high-risk loan.

The features in the dataset include various financial metrics that are used to predict the creditworthiness of the borrower.

## Technologies Used
- **Python**: Primary language for data processing and model building.
- **Pandas**: For handling and manipulating the dataset.
- **scikit-learn**: For splitting the dataset, training the logistic regression model, and evaluating model performance.
- **Jupyter Notebook**: For writing and executing the code.
- **Git/GitHub**: For version control and project submission.

## Installation
To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clonehttps://github.com/maslla100/credit-risk-classification
   ```
2. **Navigate to the Project Directory**:
   ```bash
   cd credit-risk-classification/Credit_Risk
   ```
3. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook credit_risk_classification.ipynb
   ```

## Usage
1. **Data Preprocessing**: The first step involves reading the dataset and splitting it into labels (`y`) and features (`X`). The `train_test_split` method is then used to split the dataset into training and testing sets.
2. **Model Training**: Fit a logistic regression model on the training set (`X_train` and `y_train`).
3. **Predictions**: Use the model to make predictions on the test set (`X_test`).
4. **Model Evaluation**: Evaluate the performance of the model using a confusion matrix and a classification report.
5. **Generate the Report**: Write the analysis of the model’s performance in the form of a report in `README.md`.

## Model Evaluation
The logistic regression model's performance is evaluated using the following metrics:
- **Accuracy**: Measures how often the model correctly classifies both healthy and high-risk loans.
- **Precision**: Measures how many of the predicted high-risk loans are actually high-risk.
- **Recall**: Measures how many actual high-risk loans the model correctly identifies.
- **F1-score**: A harmonic mean of precision and recall, providing a single performance measure.

### Confusion Matrix:
The confusion matrix is a table that summarizes the performance of the logistic regression model by comparing actual vs predicted outcomes.

### Classification Report:
The classification report provides detailed metrics including precision, recall, and F1-score for each class (healthy loans and high-risk loans).

### Example Confusion Matrix and Report:
After running the model, use the following lines of code to generate and review the confusion matrix and classification report:
```python
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
print(confusion_matrix(y_test, y_pred))

# Classification report
print(classification_report(y_test, y_pred))
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements
- **Bootcamp Team**: For providing the resources and guidance needed to complete this challenge.
- **scikit-learn community**: For the excellent machine learning library used to build the model.
- **Pandas**: For simplifying the process of data manipulation.
- **GitHub**: For providing a platform to share and collaborate on projects.

---

