# ML Project
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

file_name = "./Data/MoreNFLData.csv"
df = pd.read_csv(file_name)
df = df.sample(frac=1)  # random ordering of the data points
x = df.to_numpy()[:, 0:13]
# x = preprocessing.normalize(x)
y = df.to_numpy()[:, 13]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


############# Linear Regression ################
max_iter = 3000
model = Ridge(alpha=.01, max_iter=max_iter)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(r2_score(y_pred, y_test))

# print(y_test)
# print(y_pred)

# cm = confusion_matrix(y_test, y_pred, labels=[0., 1.])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0., 1.])
# disp.plot()
# plt.show()


# print("Accuracy = ", metrics.accuracy_score(y_test, y_pred))
# y_pred_prob = reg.predict_proba(x_test)[:, 1]
# auc = metrics.roc_auc_score(y_test, y_pred_prob)
# print("AUC:", round(auc, 2))
