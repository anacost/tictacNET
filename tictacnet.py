import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Cool ML here:

df = pd.read_csv('tictactoe-data.csv')

#print(df.head())
#design matrix
X = df.iloc[:, list(range(18)) + [-2]]
#print(X.head())

#doing classification:
target = df.iloc[:, list(range(18,27))]
#split into training and test dataset, 20% of dataset as test
X_train, X_test, y_train, y_test =train_test_split(X, target, test_size=0.2)
#make a model with keras( high-level API)
model = tf.keras.Sequential()
#Dense option== all neurons are connected,
# first entry is the nr of neurons in the first layer
#rectified linear unit ==> 'relu'
# 'relu' function = x, if x>= 0
#         function= 0, else
model.add(tf.keras.layers.Dense(128,activation='relu', input_dim=X.shape[1]))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(target.shape[1], activation='softmax'))
#target.shape[1] == 9
#'softmax' function which skizzes everything between 0 and 1
# to avoid overfitting, use drop-out (pick a neuron and set it to zero)
# adding a layer Dropout

#compiling the keras model:
#identify a optimizer
#identify a loss function
#identify a metrics
model.compile(optimizer= 'adam', loss= 'categorical_crossentropy',
	metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train,epochs=100,batch_size=32,
	validation_data=[X_test,y_test])
#how many times you want it to learn from the data: epochs
#batch_size: how many examples 
print('accuracy: ', model.evaluate(X_test, y_test))
model.save('tictacNET.h5')
