from __future__ import print_function
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
import argparse
import numpy as np
import pandas as pd
import sys
import os
import innvestigate
import pudb


BATCH_SIZE = 8192
EPOCHS = 10000
LEARNING_RATE = 0.0001



def categorical_accuracy_mod(y_true, y_pred):
  """ Calculates accuracy on test set.

  Args: 
      y_true: Ground truth values of test data.
      y_pred: Values predicted by the Model.

  Returns:
         Accuracy of the model.
  """
  here = np.equal(y_true, y_pred)
  return len(y_true[here])/float(len(y_true))


def train_test():
  """Lets user decide to train or test.

  Returns:
         True if user chose Train and False for test.
  """
  print("Would you like to train or test? (y for train, n for test): ")
  choice = input()
  if (choice=='y' or choice=='Y'):
    train_flag = True
  else:
    train_flag = False

  return train_flag


def load_csv(train_flag):
  """Loads data from csv file.
  Args:
      train_flag: Boolean value, True if user chose to train and False 
          if they chose to test.

  Returns: Pandas Dataframe of the csv file.
  """
  if train_flag:
    df = pd.read_csv("adult/adult_train.csv")
  else:
    df = pd.read_csv("adult/adult_test.csv")
    BATCH_SIZE = df.shape[0]
    EPOCHS = 1

  return df
    

def layer_analysis(model):
  """Implements Layer-wise relevance propagation.

  This is cutting off layers in the network to obtain heatmap 
  vectors for every layer and then merges them together to form
  a heatmap matrix/tensor. Then saves them using numpy.

  Args:
      model: The neural network to perform analysis upon.
  """
  layer_num = [2,6,10,14,18,22,26,30,34,38,42, 44]

  analyzer = innvestigate.create_analyzer("lrp.z", model)
  analysis = analyzer.analyze(x_train)
  print("analysis: "+str(analysis)+"\n\n\n")

  model.summary()
  for i in layer_num:
    print("New model ", i)
  
    new_model = Model(model.inputs, model.layers[-i].output)
    new_model.set_weights(model.get_weights())
    new_model.summary()

    analyzer = innvestigate.create_analyzer("lrp.z", new_model)
    analysis = analyzer.analyze(x_train)
    print("analysis: "+str(analysis))
    name = "out_lrp_"+str(i)
    np.save(name, analysis)




model = Sequential()

model.add(Dense(512, input_dim=108))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(8))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(2))
model.add(Activation('sigmoid'))

print(model.summary())


train_flag = train_test()

df = load_csv(train_flag)

print("Would you like to load previous saved model (y/n): ")
model_choice = input()
if (model_choice=='y' or model_choice=='Y'):
  model = load_model("checkpoints/best_model.h5")


cols = df.columns.values
last_index = (np.sum(cols.shape)-1) 
cols = np.delete(cols,last_index)
x_train = df.loc[:,cols].values
print("x_train.shape: "+str(x_train.shape))

y_train = df["income"].values
y_train_ = y_train
y_train = keras.utils.np_utils.to_categorical(y_train)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=["categorical_accuracy"])

checkpointer = ModelCheckpoint(monitor="categorical_accuracy", filepath="checkpoints/best_model.h5", verbose=True,
                                   save_best_only = True)
earlystopping = EarlyStopping(monitor="categorical_accuracy", min_delta=1e-6, patience=20, verbose=True)


if train_flag:

  model.fit(x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE, callbacks=[checkpointer, earlystopping])
else:

  pred = model.predict(x_train, batch_size=BATCH_SIZE)
  print("pred.shape: ", pred.shape)
  print("y_train.shape: ", y_train.shape)
  score = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE)
  pred_ = np.argmax(pred, axis = -1)
  unique, counts = np.unique(pred_, return_counts=True)
  print(dict(zip(unique, counts)))
  unique, counts = np.unique(y_train_, return_counts=True)
  print(dict(zip(unique, counts)))

  acc_net = categorical_accuracy_mod(y_train_, pred_)
 
  print("Acc: "+str(acc_net))
  print(model.metrics_names)
  print(score)

  layer_analysis(model)

  