import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

"""

Loan eligibility predicter

Boda Levente
Kovács Géza
Popa Vlad

"""

dataset_path = "./data/dataset.csv"
dataset_v2_path = "./data/dataset_v2.csv"

def process_data(path):
    dataset = pd.read_csv(path)

    X = dataset.iloc[:, [-4, 2, 3, -3, -2, -1]].values
    y = dataset.iloc[:, 1].values

    for row in X:
        if(row[4] == "High School or Below"):
            row[4] = 0
        if(row[4] == "BSc"):
            row[4] = 1
        if(row[4] == "Master or Above"):
            row[4] = 2

        if(row[5] == "male"):
            row[5] = 0
        if(row[5] == "female"):
            row[5] = 1

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    return X_train, X_test, y_train, y_test, X, y

# Random Forest Classifier
def random_forest_classifier(dataset, new_loan):
    clf = RandomForestClassifier(max_depth = 2, random_state = 0)
    clf.fit(dataset[4], dataset[5])

    if(new_loan == 0):
        forest_pred = clf.predict(dataset[1])
        acc_score = accuracy_score(dataset[3], forest_pred)
        print("Accuracy with Random Forest Classifier: ", acc_score)

        # Plot Predictions and Values axis
        plt.scatter(dataset[3], forest_pred)
        plt.ylabel("Predictions")
        plt.xlabel("Values")
        plt.show()

    else:
        forest_pred = clf.predict(new_loan)
        
        print("[RANDOM FOREST CLASSIFIER] Prediction of new loan - client able to pay: ")
        print(forest_pred)

    return forest_pred

# Naive Bayes classifier
def gaussian_NB_classifier(dataset, new_loan):
    stand_sclr = StandardScaler()
    x_train = stand_sclr.fit_transform(dataset[0])
    classifier = GaussianNB()
    classifier.fit(x_train, dataset[2])

    if(new_loan == 0):  
        x_test = stand_sclr.transform(dataset[1])
        y_pred = classifier.predict(x_test)
        acc_score = accuracy_score(dataset[3], y_pred)
        print("Accuracy with Naive Bayes Classifier: ", acc_score)
        
    else:
        y_pred = classifier.predict(new_loan)
        print("[NAIVE BAYES] Prediction of new loan - client able to pay: ")
        print(y_pred)
        
    return y_pred


print("\n********************************************* ")
print("Prediction with unmodified dataset: ")
print("********************************************* ")

data = process_data(dataset_path)
data_v2 = process_data(dataset_v2_path)
random_forest_classifier(data, 0)
gaussian_NB_classifier(data, 0)

print("\n\n********************************************* ")
print("Prediction with modified dataset (dataset_v2) ")
print("********************************************* ")
random_forest_classifier(data_v2, 0)
gaussian_NB_classifier(data_v2, 0)

# Due days, sum, terms, age, education, gender
new_loan = [[0, 1000, 30, 28, 1, 0]]

print("\n\n\n\n\n********************************************* ")
print("Prediction for new loan eligibility: ")
print("********************************************* ")

print("[[Due days, sum, terms, age, education, gender]]")
print(new_loan)

print("\n********************************************* ")
print("Prediction with unmodified dataset: ")
print("********************************************* ")
data = process_data(dataset_path)
# data_v2 = process_data(dataset_v2_path)
# random_forest_classifier(data, new_loan)
gaussian_NB_classifier(data, new_loan)
random_forest_classifier(data, new_loan)

print("\n\n********************************************* ")
print("Prediction with modified dataset (dataset_v2) ")
print("********************************************* ")
random_forest_classifier(data_v2, new_loan)
gaussian_NB_classifier(data_v2, new_loan)


