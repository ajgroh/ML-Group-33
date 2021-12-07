import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from keras.models import Sequential
from tensorflow.keras.layers import Dense


file_name = "./Data/MoreNFLData.csv"
df = pd.read_csv(file_name)
df = df.sample(frac=1)  # random ordering of the data points
x = df.to_numpy()[:, 0:13]
y = df.to_numpy()[:, 13]
# Scale data before applying PCA
scaling = StandardScaler()

# Use fit and transform method
scaling.fit(x)
Scaled_data = scaling.transform(x)

# Set the n_components=3
principal = PCA(n_components=3)
principal.fit(Scaled_data)
x = principal.transform(Scaled_data)

# Check the dimensions of data after PCA
print(x.shape)


plt.figure(figsize=(10, 10))
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='plasma')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

############# Logistic Regression on reduced dimmension data ################
num_iters = 1000
reg = LogisticRegression(max_iter=num_iters)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
# print(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=reg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=reg.classes_)
disp.plot()
plt.show()


print("Accuracy = ", metrics.accuracy_score(y_test, y_pred))
y_pred_prob = reg.predict_proba(x_test)[:, 1]
auc = metrics.roc_auc_score(y_test, y_pred_prob)
print("AUC:", round(auc, 2))


######## NN on reduced dimmension data #########

model = Sequential()
model.add(Dense(12, input_dim=3, activation='relu', kernel_regularizer='l2'))
model.add(Dense(8, activation='relu', kernel_regularizer='l2'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(x, y, epochs=50,
                    batch_size=10, verbose=1, validation_split=0.25)
# Plot history: accuracy
plt.plot(history.history['accuracy'], label='ACC (Training Data)')
plt.plot(history.history['val_accuracy'],
         label='ACC (Test Data)')
plt.title('Accuracy for NFL Postseason Appearance (PCA NN)')
plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
