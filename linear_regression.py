# ML Project
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# file_name = "./Data/MoreNFLData.csv"
# df = pd.read_csv(file_name)
# df = df.sample(frac=1)  # random ordering of the data points
# x = df.to_numpy()[:, 0:13]
# # x = preprocessing.normalize(x)
# y = df.to_numpy()[:, 13]


class LinearRegression:
    def __init__(self, x, y, max_iter=30000):
        self.x = x
        self.y = y
        self.max_iter = max_iter

    def main(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.25, random_state=42)
        # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        ############# Linear Regression ################

        model = Ridge(alpha=.01, max_iter=self.max_iter)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return r2_score(y_test, y_pred)

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
