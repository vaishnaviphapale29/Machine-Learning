import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

Border = "-"*40

##############################################################
# Step 1 : Load the dataset
##############################################################
print(Border)
print("Step 1 : Load the dataset")
print(Border)

DatasetPath = "iris.csv"

df = pd.read_csv(DatasetPath)

print("Dataset gets loaded succesfully....")
print("Initial entries from dataset :")
print(df.head())

##############################################################
# Step 2 : Data Analysis (EDA)
##############################################################
print(Border)
print("Step 2 : Data Analysis")
print(Border)

print("Shape of dataset : ",df.shape)
print("Column Names : ",list(df.columns))

print("Missing values(Per column)")
print(df.isnull().sum())

print("Class Distribution (Species count)")
print(df["species"].value_counts())

print("Statistical report of dataset")
print(df.describe())

##############################################################
# Step 3 : Decide Independent & Dependent variables
##############################################################
print(Border)
print("Step 3 : Decide Independent & Dependent variables")
print(Border)

# x : Indenpendent variabls / Features
# y : denpendent variabls / Labels

feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

X = df[feature_cols]
Y = df["species"]

print("X shape : ",X.shape)
print("Y shape : ",Y.shape)

##############################################################
# Step 4 : Visualization of dataset
##############################################################
print(Border)
print("Step 4 : Visualization dataset")
print(Border)

# Scatterplot
plt.figure(figsize=(7,5))

for sp in df["species"].unique():
    temp = df[df["species"] == sp]
    plt.scatter(temp["petal length (cm)"], temp["petal width (cm)"], label = sp) 

plt.title("Iris : Petal length vs Petal width")
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")

plt.legend()
plt.grid(True)
plt.show()

##############################################################
# Step 5 : Split the dataset for training and testing
##############################################################
print(Border)
print("Step 5 : Split the dataset for training and testing")
print(Border)

# Test size = 20%
# Train size = 80%

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("Data splitting activity done : ")

print("X - Independent : ",X.shape)           # (150,4)
print("Y - Dependent : ",Y.shape)             # (150, )

print("X_train : ", X_train.shape)            # (120,4)
print("X_test : ", X_test.shape)              # (30,4)

print("Y_train : ", Y_train.shape)            # (120,)
print("Y_test : ", Y_test.shape)              # (30, )

##############################################################
# Step 6 : Build the model
##############################################################
print(Border)
print("Step 6 : Build the model")
print(Border)

print("We are going to use DecisionTreeClassifier")

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

print("Model succesfully created : ",model)

##############################################################
# Step 7 : Train the model
##############################################################
print(Border)
print("Step 7 : Train the model")
print(Border)

model.fit(X_train,Y_train)

print("Model training completed")

##############################################################
# Step 8 : Test / Evaluate the model
##############################################################
print(Border)
print("Step 8 : Test / Evaluate the model")
print(Border)

Y_pred = model.predict(X_test)

print("Model evaluation (testing) complete")

print(Y_pred.shape)

print("Expected answer : ")
print(Y_test)

print("Predicted answer : ")
print(Y_pred)

##############################################################
# Step 9 : Evaluate the model performance
##############################################################
print(Border)
print("Step 9 : Evaluate the model performance")
print(Border)

accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy of model is : ",accuracy*100)

cm = confusion_matrix(Y_test,Y_pred)
print("Confusion matrix : ")
print(cm)

print("Classification report")
print(classification_report(Y_test,Y_pred))

##############################################################
# Step 10 : Plot confusion matrix
##############################################################
print(Border)
print("Step 10 : Plot confusion matrix")
print(Border)

data = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
data.plot()
plt.title("Confusion matrix of Iris dataset")
plt.show()

##############################################################
# Step 11 : Plot Decision Tree
##############################################################
print(Border)
print("Step 11 : Plot Decision Tree")
print(Border)

plt.figure(figsize=(18,10))

plot_tree(
    model,
    feature_names=feature_cols,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree - Iris Dataset")
plt.show()