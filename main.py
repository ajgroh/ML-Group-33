# ML Project
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from NN import NN
from svm import SVM
from logistic_regression import Logistic_Regression
from linear_regression import LinearRegression


file_name = "./Data/MoreNFLData.csv"
df = pd.read_csv(file_name)
df = df.sample(frac=1)  # random ordering of the data points
x = df.to_numpy()[:, 0:13]

scaling = preprocessing.StandardScaler()

# Use fit and transform method
scaling.fit(x)
Scaled_data = scaling.transform(x)

y = df.to_numpy()[:, 13]

#### ---- Neural Network ----- ####
neural = NN(Scaled_data, y)
historyNN = neural.main()
# neural.plot(historyNN)
test_nn_acc = historyNN.history['val_accuracy']
print("Nueral Network Accuracy: ", test_nn_acc[len(test_nn_acc) - 1])

#### ---- Support Vector Machine ----- ####
svm = SVM(Scaled_data, y)
svmModel = svm.main()

#### ---- Logistic Regression ----- ####
lr = Logistic_Regression(x, y, 10000)
reg, x_test, y_test = lr.main()
y_pred = reg.predict(x_test)
print("Logistic Regression Accuracy = ",
      metrics.accuracy_score(y_test, y_pred))

#### ---- Linear Regression ----- ####
lir = LinearRegression(x, y, 30000)
lir_r2 = lir.main()
print("Linear Regression R2 Score = ", lir_r2)
