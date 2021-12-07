# ML Project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# file_name = "./Data/MoreNFLData.csv"
# df = pd.read_csv(file_name)
# df = df.sample(frac=1)  # random ordering of the data points
# x = df.to_numpy()[:, 0:13]

# # Scale data before applying PCA
# scaling = preprocessing.StandardScaler()

# # Use fit and transform method
# scaling.fit(x)
# Scaled_data = scaling.transform(x)

# y = df.to_numpy()[:, 13]


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
        num_iters = 10000
        reg = LogisticRegression(max_iter=num_iters)
        reg.fit(x_train, y_train)

        # print(type(reg.classes_))

        return reg, x_test, y_test

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
