#ML Project
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

file_name = "./Data/NFLData.csv"
df = pd.read_csv(file_name)
df = df.sample(frac=1) #random ordering of the data points
x = df.to_numpy()[:, 0:12]

# Scale data before applying PCA
scaling=preprocessing.StandardScaler()

# Use fit and transform method
scaling.fit(x)
Scaled_data=scaling.transform(x)

y = df.to_numpy()[:, 12]
x_train, x_test, y_train, y_test = train_test_split(Scaled_data, y, test_size=0.25, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)




############# Logistic Regression ################
num_iters = 1000
reg = LogisticRegression(max_iter = num_iters)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
# print(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=reg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=reg.classes_)
disp.plot()
plt.show()


print("Accuracy = ", metrics.accuracy_score(y_test, y_pred))
y_pred_prob = reg.predict_proba(x_test)[:,1]
auc = metrics.roc_auc_score(y_test, y_pred_prob)
print("AUC:", round(auc, 2))
