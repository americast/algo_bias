from __future__ import print_function
import keras
from keras.models import Sequential, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
import argparse
import numpy as np
import pandas as pd
import sys
import os
# import pudb

BATCH_SIZE = 8192
EPOCHS = 100000
LEARNING_RATE = 0.0001

model = Sequential()
model.add(Dense(256, input_dim=166))
model.add(Activation('selu'))


model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('selu'))
# model.add(Dropout(0.4))


model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('selu'))
# model.add(Dropout(0.4))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('selu'))
# model.add(Dropout(0.4))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('selu'))
# model.add(Dropout(0.4))


model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('selu'))
# model.add(Dropout(0.4))


model.add(Dense(16))
model.add(BatchNormalization())
model.add(Activation('selu'))
# model.add(Dropout(0.4))

model.add(Dense(8))
model.add(BatchNormalization())
model.add(Activation('selu'))
# model.add(Dropout(0.4))

model.add(Dense(3))
model.add(Activation('sigmoid'))
print(model.summary())

print("Would you like to restore a previously saved model? (y/n)")
choice = input()

if (choice=='y' or choice=='Y'):
  #path = input("Enter path: ")
  model = load_model("checkpoints/model.h5")

print("\n")


train_flag = True

print("Train? (y for train, n for test)")
choice = input()
if (choice =='n' or choice=='N'):
  df = pd.read_csv("data/out-test.csv")
  BATCH_SIZE = df.shape[0]
  EPOCHS = 1
  train_flag = False
  
else:
  df = pd.read_csv("data/out-train.csv")

def categorical_accuracy_mod(y_true, y_pred):
  here = np.equal(y_true, y_pred)
  return len(y_true[here])/float(len(y_true))

cols = df.columns.values
cols = np.delete(cols,[1])
x_train = df.loc[:,cols].values

y_train = df["decile_score"].values
y_train_ = y_train
y_train = keras.utils.np_utils.to_categorical(y_train)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=["categorical_accuracy"])

checkpointer = ModelCheckpoint(monitor="categorical_accuracy", filepath="checkpoints/model.h5", verbose=True,
                                   save_best_only = True)
earlystopping = EarlyStopping(monitor="categorical_accuracy", min_delta=1e-6, patience=20, verbose=True)


if train_flag:
  model.fit(x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE, callbacks=[checkpointer, earlystopping])
else:
  score = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE)
  pred = model.predict(x_train, batch_size=BATCH_SIZE)
  pred_ = np.argmax(pred, axis = -1)
  print("Hello!: "+str(pred.shape))
  print("Hello!: "+str(pred_.shape))
  unique, counts = np.unique(pred_, return_counts=True)
  print(dict(zip(unique, counts)))
  unique, counts = np.unique(y_train_, return_counts=True)
  print(dict(zip(unique, counts)))

  acc_net = categorical_accuracy_mod(y_train_, pred_)
  # print("no. correct: "+str(len(y_train[bool_res])))
  # print("total no.: "+str(len(y_train)))
  print("Acc: "+str(acc_net))
  print(model.metrics_names)
  print(score)
  # pu.db


