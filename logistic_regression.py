# ML Project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Logistic_Regression():
    def __init__(self, x, y, max_iter=10000):
        self.x = x
        self.y = y
        self.max_iter = max_iter

    def main(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.25, random_state=42)
        # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        ############# Logistic Regression ################
        reg = LogisticRegression(max_iter=self.max_iter)
        reg.fit(x_train, y_train)

        y_pred = reg.predict(x_test)
        # print(type(reg.classes_))

        # return reg, x_test, y_test
        return metrics.accuracy_score(y_test, y_pred)

    def plot(self, model, x_test, y_test):

        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()
        plt.show()

        print("Accuracy = ", metrics.accuracy_score(y_test, y_pred))
        y_pred_prob = model.predict_proba(x_test)[:, 1]
        auc = metrics.roc_auc_score(y_test, y_pred_prob)
        print("AUC:", round(auc, 2))
