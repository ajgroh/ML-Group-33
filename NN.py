# ML Project
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import tensorflow
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


file_name = "./Data/MoreNFLData.csv"
df = pd.read_csv(file_name)
df = df.sample(frac=1)  # random ordering of the data points
x = df.to_numpy()[:, 0:12]

# Scale data before applying NN
scaling = preprocessing.StandardScaler()

# Use fit and transform method
scaling.fit(x)
Scaled_data = scaling.transform(x)

y = df.to_numpy()[:, 12]
# x_train, x_test, y_train, y_test = train_test_split(
#     Scaled_data, y, test_size=0.25, random_state=42)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


##### Create the Model #####
model = Sequential()
model.add(Dense(12, input_dim=12, activation='relu', kernel_regularizer='l2'))
model.add(Dense(8, activation='relu', kernel_regularizer='l2'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(Scaled_data, y, epochs=50,
                    batch_size=10, verbose=1, validation_split=0.25)


# Plot history: accuracy
plt.plot(history.history['accuracy'], label='ACC (Training Data)')
plt.plot(history.history['val_accuracy'],
         label='ACC (Test Data)')
plt.title('Accuracy for NFL Postseason Appearance')
plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
