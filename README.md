# HappyNote


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import LSTM, Dropout, Dense, Activation
# from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import datetime

import tensorflow as tf



data = pd.read_csv('data/random_data.csv')

#df = data[data['High'].notnull()]
#data = data[data['High'].notnull()]

#print(data)


result = np.array(data)

# split train and test data
row = int(round(result.shape[0] * 0.9))

# train = result[:row, :]
# np.random.shuffle(train)

from sklearn.preprocessing import LabelEncoder


target1=data['INPUT1'].values.tolist()
target2=data['INPUT2'].values.tolist()
target3=data['INPUT3'].values.tolist()
length = len(target1)
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
label_encoder3 = LabelEncoder()
Y1 = np.array(label_encoder1.fit_transform(target1)).reshape((length, 1))
Y2 = np.array(label_encoder2.fit_transform(target2)).reshape((length, 1))
Y3 = np.array(label_encoder3.fit_transform(target3)).reshape((length, 1))

result = np.array(data)[:,:3]

result = np.concatenate([Y1,Y2,Y3, result], axis=1).astype('float32')
train = result[:row, :]
# np.random.shuffle(train)

x_train = train[:, :3]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, 3:]
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

print(row)
x_test = result[row:, :3]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, 3:]
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

tf.keras.optimizers
tf.keras.losses.Loss
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(13, return_sequences=True, input_shape=(3,1)))
model.add(tf.keras.layers.Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='RMSprop')
model.summary()
# tf.keras.optimizers
model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
#    batch_size=50,
    epochs=10)

pred = model.predict(x_test)

# fig = plt.figure(facecolor='white', figsize=(20, 10))
# ax = fig.add_subplot(111)
# ax.plot(y_test[:,1:], label='True')
# ax.plot(pred[:,1:], label='Prediction')
# ax.legend()
# plt.show()

print(x_test[:10])
print(y_test[:10])
print(pred[:10])
