import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import tensorflow
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import metrics
from sklearn import preprocessing, svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class SVM:
    # file_name = "./Data/MoreNFLData.csv"
    # df = pd.read_csv(file_name)
    # df = df.sample(frac=1)  # random ordering of the data points
    # x = df.to_numpy()[:, 0:13]

    # # Scale data before applying NN
    # scaling = preprocessing.StandardScaler()

    # # Use fit and transform method
    # scaling.fit(x)
    # Scaled_data = scaling.transform(x)

    # y = df.to_numpy()[:, 13]
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def main(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.25, random_state=42)

        ###### Create and fit the model #######
        # hyperparameter for regularization
        C = [0.01, 0.1, 0.5, 1.0, 10, 50, 100]
        d = 3  # hyperparameter for degree of polynomial
        maxAcc = 0
        maxAccC = 0
        maxAuc = 0
        for c in C:
            clf = svm.SVC(C=c, degree=d, probability=True)
            clf = clf.fit(self.x_train, self.y_train)

            y_pred = clf.predict(self.x_test)

            # print("Accuracy = ", metrics.accuracy_score(y_test, y_pred))
            y_pred_prob = clf.predict_proba(self.x_test)[:, 1]
            auc = metrics.roc_auc_score(self.y_test, y_pred_prob)
            acc = metrics.accuracy_score(self.y_test, y_pred)
            # print("AUC:", round(auc, 2))
            if acc > maxAcc:
                maxAcc = acc
                maxAccC = c
                maxAuc = auc
        print("Max SVM Accuracy ", maxAcc, " with C value ",
              maxAccC, ". Corresponding AUC: ", maxAuc)
        return maxAcc

        ##### Making hyperparameter plot ######
    def plot(self):
        # hyperparameter for regularization
        C = [0.01, 0.1, 0.5, 1.0, 10, 50, 100]
        d = 3  # hyperparameter for degree of polynomial
        acc = []
        auc_list = []
        for c in C:
            clf = svm.SVC(C=c, degree=d, probability=True)
            clf.fit(self.x_train, self.y_train)

            y_pred = clf.predict(self.x_test)
            cm = confusion_matrix(self.y_test, y_pred, labels=clf.classes_)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=clf.classes_)
            disp.plot()
            plt.show()

            # print("Accuracy = ", metrics.accuracy_score(y_test, y_pred))
            y_pred_prob = clf.predict_proba(self.x_test)[:, 1]
            auc = metrics.roc_auc_score(self.y_test, y_pred_prob)
            # print("AUC:", round(auc, 2))
            acc.append(metrics.accuracy_score(self.y_test, y_pred))
            auc_list.append(round(auc, 2))
        # Plot: C
        C = np.log10(np.array(C))
        plt.plot(C, acc, label='Accuracy (test data)')
        plt.plot(C, auc_list,
                 label='AUC Value')
        plt.title('ACC/AUC for NFL Postseason Appearance ')
        plt.ylabel('ACC value')
        plt.xlabel('C value (Regularization parameter log10 scale)')
        plt.legend(loc="upper left")
        plt.show()

        # # Plot: d
        # C = 10
        # D = [1, 2, 3, 5, 10, 50]
        # acc_d = []
        # auc_list_d = []

# for d in D:
#     clf = svm.SVC(C=C, degree=d, probability=True)
#     clf.fit(x_train, y_train)

#     y_pred = clf.predict(x_test)
#     cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
#     disp = ConfusionMatrixDisplay(
#         confusion_matrix=cm, display_labels=clf.classes_)
#     disp.plot()
#     plt.show()

#     # print("Accuracy = ", metrics.accuracy_score(y_test, y_pred))
#     y_pred_prob = clf.predict_proba(x_test)[:, 1]
#     auc = metrics.roc_auc_score(y_test, y_pred_prob)
#     # print("AUC:", round(auc, 2))
#     acc_d.append(metrics.accuracy_score(y_test, y_pred))
#     auc_list_d.append(round(auc, 2))
# plt.plot(D, acc_d, label='Accuracy (test data)')
# plt.plot(D, auc_list_d,
#          label='AUC Value')
# plt.title('ACC/AUC for NFL Postseason Appearance ')
# plt.ylabel('ACC value')
# plt.xlabel('D value (Degree of kernel)')
# plt.legend(loc="upper left")
# plt.show()
