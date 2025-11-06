# -------------------------------------------------------------------------
# AUTHOR: Joseline Ly
# FILENAME: decision_tree.py
# SPECIFICATION: Creates decision trees for multiple test datasets.
# FOR: CS 4440 (Data Mining) - Assignment #3
# TIME SPENT: 8 hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    # Transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    # Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    # be converted to a float.
    # X =
    for row in data_training:
        refund = 1 if row[0] == 'Yes' else 0
        single = 1 if row[1] == 'Single' else 0
        divorced = 1 if row[1] == 'Divorced' else 0
        married = 1 if row[1] == 'Married' else 0
        # For taxable income in row[2], remove the k to retain numeric value and convert to float
        taxable_income = float(row[2].replace('k', ''))
        X.append([refund, single, divorced, married, taxable_income])

    # Transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    # --> add your Python code here
    # Y =
    for row in data_training:
        Y.append(1 if row[3] == 'Yes' else 0)

    # Loop your training and test tasks 10 times here
    for i in range (10):
       correct_predictions = 0

       # Fitting the decision tree to the data by using Gini index and no max_depth
       clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
       clf = clf.fit(X, Y)

       # Plotting the decision tree
       tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
       plt.show()

       # Read the test data and add this data to data_test NumPy
       # --> add your Python code here
       # data_test =
       data_test = pd.read_csv('cheat_test.csv', sep=',', header=0).to_numpy()

       for data in data_test:
           # Transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           # class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           # --> add your Python code here
            refund = 1 if data[1] == 'Yes' else 0 # type: ignore
            taxable_income = float(data[3].replace('k',''))

           # Compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
           # --> add your Python code here
            class_predicted = clf.predict([[refund, single, divorced, married, taxable_income]])[0] # type: ignore
            if class_predicted == (1 if data[3] == 'Yes' else 0):
               correct_predictions += 1

       # Find the average accuracy of this model during the 10 runs (training and test set)
       # --> add your Python code here
            accuracy = correct_predictions / len(data_test)  # type: ignore

    # Print the accuracy of this model during the 10 runs (training and test set).
    # Your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    # --> add your Python code here
    print(f'Final accuracy when training on {ds}: {accuracy}')
