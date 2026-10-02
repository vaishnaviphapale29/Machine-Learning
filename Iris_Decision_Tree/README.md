# Iris Dataset – Decision Tree Classification

## Project Overview

This project demonstrates a Machine Learning classification task using the **Decision Tree Classifier** on the Iris dataset.

The model predicts the species of an Iris flower using its sepal and petal measurements.

## Objective

The objective of this project is to build and evaluate a Decision Tree model for classifying different Iris flower species.

## Dataset

The dataset contains four input features:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

**Target Variable:** Species

## Project Workflow

1. Load the Iris dataset using Pandas.
2. Perform basic data analysis.
3. Check missing values and class distribution.
4. Select independent and dependent variables.
5. Visualize the data using a scatter plot.
6. Split the dataset into 80% training and 20% testing data.
7. Build and train the Decision Tree model.
8. Generate predictions for the test data.
9. Evaluate the model using accuracy, confusion matrix, and classification report.
10. Visualize the confusion matrix.
11. Visualize the trained Decision Tree.

## Model Configuration

```text
Algorithm      : Decision Tree Classifier
Criterion      : Gini
Maximum Depth  : 5
Random State   : 42
Training Data  : 80%
Testing Data   : 20%
```

## Model Evaluation

The model is evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-Score

The Decision Tree is also visualized to understand the model's decision-making process.

## Installation

Install the required libraries using:

```bash
pip install -r requirements.txt
```

## How to Run

Run the Python program using:

```bash
python Iris_DecisionTreeClassifier.py
```

The program displays the dataset analysis, scatter plot, model accuracy, confusion matrix, classification report, and Decision Tree visualization.
