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
from pca import OurPCA

nnA = []
svmA = []
lorA = []
lirA = []
plorA = []
pnnA = []

for i in range(10):

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
    nnA.append(test_nn_acc[len(test_nn_acc) - 1])
    print("Nueral Network Accuracy: ", test_nn_acc[len(test_nn_acc) - 1])

    #### ---- Support Vector Machine ----- ####
    svm = SVM(Scaled_data, y)
    svmAcc = svm.main()
    svmA.append(svmAcc)

    #### ---- Logistic Regression ----- ####
    lr = Logistic_Regression(x, y, 10000)
    lr_acc = lr.main()
    lorA.append(lr_acc)
    print("Logistic Regression Accuracy = ", lr_acc)

    #### ---- Linear Regression ----- ####
    lir = LinearRegression(x, y, 30000)
    lir_r2 = lir.main()
    lirA.append(lir_r2)
    print("Linear Regression R2 Score = ", lir_r2)

    #### ---- Principle Component Analysis ----- ####
    pca = OurPCA(x, y)
    pcaLRAcc = pca.main()
    PCAhistory = pca.NNmain()
    plorA.append(pcaLRAcc)
    print("Logistic Regression With PCA Accuracy = ", pcaLRAcc)
    test_pca_acc = PCAhistory.history['val_accuracy']
    pnnA.append(test_pca_acc[len(test_pca_acc) - 1])
    print("Nueral Network with PCA Accuracy: ",
          test_pca_acc[len(test_pca_acc) - 1])

nnA = np.array(nnA)
svmA = np.array(svmA)
lorA = np.array(lorA)
lirA = np.array(lirA)
plorA = np.array(plorA)
pnnA = np.array(pnnA)

print("NN Acc Avg: ", np.mean(nnA), " Var: ", np.var(nnA))
print("SVM Acc Avg: ", np.mean(svmA), " Var: ", np.var(svmA))
print("Log Reg Acc Avg: ", np.mean(lorA), " Var: ", np.var(lorA))
print("Lin Reg R^2 Avg: ", np.mean(lirA), " Var: ", np.var(lirA))
print("Log Reg w PCA Acc Avg: ", np.mean(plorA), " Var: ", np.var(plorA))
print("NN Acc w PCA Avg: ", np.mean(pnnA), " Var: ", np.var(pnnA))
