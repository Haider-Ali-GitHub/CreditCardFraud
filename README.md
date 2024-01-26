
# Credit Card Fraud Detection Project

## Introduction

This project aims to detect fraudulent transactions in credit card data using various machine learning models. The project involves exploratory data analysis (EDA), preprocessing, model training, prediction, and evaluation. Key models used include Decision Tree, K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machine (SVM), Random Forest, and XGBoost. The dataset used is a comprehensive collection of credit card transactions, with a focus on distinguishing between legitimate and fraudulent activities.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Features](#features)
5. [Dependencies](#dependencies)
6. [Configuration](#configuration)
7. [Documentation](#documentation)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)



## Installation

To use this project, you need to install Python and the following libraries:
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- XGBoost

You can install these packages using pip:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost
```

## Usage

To run the project, import the script and call the main function. Ensure the credit card dataset (`creditcard.csv`) is available in the project directory.

## Features

- **Exploratory Data Analysis (EDA):** Analyze the dataset to understand the distribution of legitimate and fraudulent transactions.
- **Data Preprocessing:** Includes scaling of features and splitting data into training and testing sets.
- **Model Training:** Train multiple models like Decision Tree, KNN, Logistic Regression, SVM, Random Forest, and XGBoost.
- **Predictions:** Use the trained models to predict fraudulent transactions.
- **Evaluation:** Evaluate model performance using accuracy and F1 score metrics.
- **Confusion Matrix Visualization:** Plot and save confusion matrices for each model.

## Dependencies

This project requires the following Python libraries:
- `itertools`
- `pandas`
- `numpy`
- `matplotlib`
- `sklearn`
- `xgboost`

## Configuration

No additional configuration is required apart from the standard setup of Python and the necessary libraries.

## Documentation

For further documentation, refer to the official documentation of each library used:
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)

## Examples

Example of using the Decision Tree model:

```python
# Decision Tree
decision_tree = DecisionTreeClassifier(max_depth=4, criterion='entropy')
decision_tree.fit(features_train, labels_train)
decision_tree_predictions = decision_tree.predict(features_test)
```

## Troubleshooting

If you encounter any issues with library versions or data loading, ensure you have the correct versions of the libraries installed and the dataset is correctly placed in the project directory.

