import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

dataset_path = "./data/dataset.csv"
dataset_v2_path = "./data/dataset_v2.csv"

# data = read_data()
# print(data.head())
# print(split_by_class(data).head)

def process_data(path):
    dataset = pd.read_csv(path)
    # new_data = pd.read_csv("./data/new_stuff.csv")
    X = dataset.iloc[:, [-4, 2, 3, -3, -2, -1]].values
    y = dataset.iloc[:, 1].values

    # X_2 = dataset_2.iloc[:, [-4, 2, 3, -3, -2, -1]].values
    # y_2 = dataset_2.iloc[:, 1].values

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

    # for row in X_2:
    #     if(row[4] == "High School or Below"):
    #         row[4] = 0
    #     if(row[4] == "BSc"):
    #         row[4] = 1
    #     if(row[4] == "Master or Above"):
    #         row[4] = 2

    #     if(row[5] == "male"):
    #         row[5] = 0
    #     if(row[5] == "female"):
    #         row[5] = 1

        
    # print(X)
    # Splitting the dataset into the Training set and Test set
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    return X_train, X_test, y_train, y_test, X, y
    X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, test_size = 0.20, random_state = 0)

# Random Forest Classifier
def random_forest_classifier(dataset):
    clf = RandomForestClassifier(max_depth = 2, random_state = 0)
    clf.fit(dataset[4], dataset[5])

    forest_pred = clf.predict(dataset[1])
    acc_score = accuracy_score(dataset[3], forest_pred)

    print("Accuracy with Random Forest Classifier: ", acc_score)
    return forest_pred

# print("forest pred = ", ac)

# clf_2 = RandomForestClassifier(max_depth = 2, random_state = 0)
# clf_2.fit(X_2, y_2)

# forest_pred_2 = clf_2.predict(X_2_test)
# ac = accuracy_score(y_2_test, forest_pred_2)
# print("forest pred_2 = ", ac)

# Feature Scaling

#return X_train, X_test, y_train, y_test, X, y

def gaussian_NB_classifier(dataset):

    stand_sclr = StandardScaler()

    x_train = stand_sclr.fit_transform(dataset[0])
    x_test = stand_sclr.transform(dataset[1])
    # dataset.X_train = stand_sclr.fit_transform(dataset.X_train)
    # dataset.X_test = stand_sclr.transform(dataset.X_test)

    # Training the Naive Bayes model on the Training set
    classifier = GaussianNB()
    classifier.fit(x_train, dataset[2])
    y_pred = classifier.predict(x_test)

    acc_score = accuracy_score(dataset[3], y_pred)
    print("Accuracy with Naive Bayes Classifier: ", acc_score)
    return y_pred

print("\n********************************************* ")
print("Prediction with unmodified dataset: ")
print("********************************************* \n")
data = process_data(dataset_path)
data_v2 = process_data(dataset_v2_path)
random_forest_classifier(data)
gaussian_NB_classifier(data)

print("\n\n********************************************* ")
print("Prediction with modified dataset (dataset_v2) ")
print("********************************************* ")
random_forest_classifier(data_v2)
gaussian_NB_classifier(data_v2)
# Predicting the Test set results

# due days, sum, terms, age, education, gender
# ex = [[-10, 1000, 30, 28, 1, 0]]


# y_new_pred = classifier.predict(ex)
# new_forest_pred = clf.predict(ex)
# print(y_new_pred)
# print(new_forest_pred)


# print("\n\n\n *************************************** \n\n\n")

# # Accuracy score

# ac = accuracy_score(y_test,y_pred)
# print(ac)

# plt.scatter(y_test, y_pred)
# plt.ylabel("Predictions")
# plt.xlabel("Values")

# plt.show()

