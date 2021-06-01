import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.sparse import data
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn import tree

warnings.filterwarnings("ignore")

"""

@title          Loan eligibility predicter

@authors        Boda Levente
                Kovács Géza
                Popa Vlad

"""

dataset_path = "./data/dataset.csv"
dataset_v2_path = "./data/dataset_v2.csv"

"""
@brief          Reads data from dataset, processes it and creates training- and testsets
@param path     Path to dataset (.csv file)

@return tuple = (X_train, X_test, y_train, y_test, X, y)
"""
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

"""
@brief                  Random Forest Classifier to predict loan eligibility
@param dataset          Data returned from process_data() function
@param new_loan         New entry of data in [[Due days, sum, terms, age, education, gender]] format

@return forest_pred     Predicts if loan will be paid back or not [0, 1, 'COLLECTION', 'COLLECTION_PAIDOFF', 'PAIDOFF']
"""
def random_forest_classifier(dataset, new_loan):
    clf = RandomForestClassifier(random_state = 0)
    X = dataset[4]
    y = dataset[5]
    X_test = dataset[1]
    y_test = dataset[3]
    clf.fit(X, y)

    if(new_loan == 0):
        forest_pred = clf.predict(X_test)
        acc_score = accuracy_score(y_test, forest_pred)
        print("Accuracy with Random Forest Classifier: ", acc_score)

        # Plot Predictions and Values axis
        # plt.scatter(dataset[3], forest_pred)
        # plt.ylabel("Predictions")
        # plt.xlabel("Values")
        # plt.show()
    else:
        forest_pred = clf.predict(new_loan)
        
        print("[RANDOM FOREST CLASSIFIER] Prediction of new loan - client able to pay: ")
        print(forest_pred)

    return forest_pred

"""
@brief             Gaussian Naive Bayes to predict loan eligibility
@param dataset     Data returned from process_data() function
@param new_loan    New entry of data in [[Due days, sum, terms, age, education, gender]] format

@return y_pred     Predicts if loan will be paid back or not [0, 1, 'COLLECTION', 'COLLECTION_PAIDOFF', 'PAIDOFF']
"""
def gaussian_NB_classifier(dataset, new_loan):
    stand_sclr = StandardScaler()
    x_train = stand_sclr.fit_transform(dataset[0])
    y_train = dataset[2]
    y_test  = dataset[3]
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)

    if(new_loan == 0):  
        x_test = stand_sclr.transform(dataset[1])
        y_pred = classifier.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred)
        print("Accuracy with Naive Bayes Classifier: ", acc_score)
        
    else:
        y_pred = classifier.predict(new_loan)
        print("[NAIVE BAYES] Prediction of new loan - client able to pay: ")
        print(y_pred)
        
    return y_pred

"""
@brief             Return i-th column of a matrix
@param matrix      Matrix we'd like to get the i-th column of
@param i           Index of the column we'd like to return

@return row[i]     i-th column of the matrix
"""
def column(matrix, i):
    return [row[i] for row in matrix]

"""
@brief             Linear Regression
@param dataset     Data returned from process_data() function
@param new_loan    New entry of data in [[Due days, sum, terms, age, education, gender]] format

@return y_pred     Chance of debt to be paid back in time
"""
def linear_regression(dataset, new_loan):
    from sklearn.linear_model import LinearRegression
    
    print("Using Linear Regression: ")
    # We'll assign some weights to each
    ## PAIDOFF - 1
    ## UNPAID / COLLECTION - 0
    ## COLLECTION_PAIDOFF - 0.5 (late payment)
    x_train = dataset[0]
    x_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]
    y       = dataset[5]
    y_train = np.where(y_train == 'PAIDOFF', 1, y_train)
    y_train = np.where(y_train == 'UNPAID', 0, y_train)
    y_train = np.where(y_train == 'COLLECTION_PAIDOFF', 0.5, y_train)
    y_train = np.where(y_train == 'COLLECTION', 0, y_train)

    # Train dataset
    reg = LinearRegression().fit(x_train, y_train)
    # Accuracy score
    print("Accuracy score = ", reg.score(x_train, y_train))

    # Coefficients describe which relationships are more/less significant
    print("Coefficients = ", reg.coef_)

    # Intercept indicates the location where the slope intersects an axis
    print("Intercept = ", reg.intercept_)

    y_pred = reg.predict(x_test)

    # We'll assign some weights to each
    ## PAIDOFF - 1
    ## UNPAID / COLLECTION - 0
    ## COLLECTION_PAIDOFF - 0.5 (late payment)
    y_test = np.where(y_test == 'PAIDOFF', 1, y_test)
    y_test = np.where(y_test == 'UNPAID', 0, y_test)
    y_test = np.where(y_test == 'COLLECTION_PAIDOFF', 0.5, y_test)
    y_test = np.where(y_test == 'COLLECTION', 0, y_test)

    days = column(x_test, 0)
    days_array = np.array(days, dtype='float')
    y_test_array = np.array(y_test, dtype='float')
    y_pred_array = np.array(y_pred, dtype='float')

    # Plotting points, labels
    fig = plt.figure()
    fig.canvas.set_window_title("Linear Regression")
    plt.title("Probability of paying back by days")
    plt.xlabel("Pays in time")
    plt.ylabel("Probability of paying back")
    plt.plot(days, y_pred, 'o', color='blue')
    plt.plot(days, y_test_array, 'o', color='red')

    # "Decision" line
    plt.axline((0,0), (0,1.1), linestyle='--')

    # Best fit line
    ## Where:
    ## m = slope
    ## b = intercept
    m , b = np.polyfit(days_array, y_pred_array, 1)
    plt.plot(days_array, m * days_array + b)
    plt.show()

    # Print probability of new loanee to pay back in time
    if (new_loan != 0):
        y_pred = reg.predict(new_loan)
        print("Prediction for new loan = %.2f" % (y_pred[0] * 100), "% chance of paying back")

    return y_pred

"""
@brief             Decision Tree with Classifier
@param data        Data returned from process_data() function
"""
def decision_tree_classifier(data):
    X = data[4]
    y = data[5]
    fig = plt.figure(figsize = [16, 10], dpi = 115)
    plt.title("Decision Tree with Classifier")
    fig.canvas.set_window_title("Decision Tree with Classifier")
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(X, y)
    tree.plot_tree(decision_tree)
    plt.show()

"""
@brief             Decision Tree with Regressor
@param data        Data returned from process_data() function
"""
def decision_tree_regressor(data):
    fig = plt.figure(figsize = [16, 10], dpi = 115)
    plt.title("Decision Tree with Regressor")
    decision_tree = tree.DecisionTreeRegressor()
    fig.canvas.set_window_title("Decision Tree with Regressor")
    X = data[4]
    y = data[5]
    y = np.where(y == 'PAIDOFF', 1, y)
    y = np.where(y == 'UNPAID', 0, y)
    y = np.where(y == 'COLLECTION_PAIDOFF', 0.5, y)
    y = np.where(y == 'COLLECTION', 0, y)
    decision_tree = decision_tree.fit(X, y)
    tree.plot_tree(decision_tree)
    plt.show()

def main():
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

    # Create new_loan object
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
    gaussian_NB_classifier(data, new_loan)
    random_forest_classifier(data, new_loan)

    print("\n\n********************************************* ")
    print("Prediction with modified dataset (dataset_v2) ")
    print("********************************************* ")

    random_forest_classifier(data_v2, new_loan)
    gaussian_NB_classifier(data_v2, new_loan)

    print("\n\n********************************************* ")
    print("Prediction using Linear Regression: ")
    print("********************************************* ")
    # linear_regression(data, 0)
    linear_regression(data, new_loan)

    # Decision tree with Classifier
    decision_tree_classifier(data)

    # Decision tree with Regressor
    decision_tree_regressor(data)

if __name__ == "__main__":
    main()