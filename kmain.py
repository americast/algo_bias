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


print("Would you like to restore a previously saved model? (y/n)")
choice = input()

if (choice=='y' or choice=='Y'):
  #path = input("Enter path: ")
  model = load_model("checkpoints/best_model.h5")

print("\n")


train_flag = True

print("Train? (y for train, n for test)")
choice = input()
if (choice =='n' or choice=='N'):
  df = pd.read_csv("adult/adult_test.csv")
  BATCH_SIZE = df.shape[0]
  EPOCHS = 1
  train_flag = False
  
else:
  df = pd.read_csv("adult/adult_train.csv")

def categorical_accuracy_mod(y_true, y_pred):
  here = np.equal(y_true, y_pred)
  return len(y_true[here])/float(len(y_true))

cols = df.columns.values
last_index = (np.sum(cols.shape)-1) 
cols = np.delete(cols,last_index)
x_train = df.loc[:,cols].values
print("x_train.shape: "+str(x_train.shape))

y_train = df["income"].values
y_train_ = y_train
y_train = keras.utils.np_utils.to_categorical(y_train)
# if not train_flag:
#   y_train = np.repeat([[1,0]], y_train.shape[0], axis = 0)

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
  # print("no. correct: "+str(len(y_train[bool_res])))
  # print("total no.: "+str(len(y_train)))
  print("Acc: "+str(acc_net))
  print(model.metrics_names)
  print(score)

  analyzer = innvestigate.create_analyzer("lrp.z", model)
  analysis = analyzer.analyze(x_train)
  print("analysis: "+str(analysis)+"\n\n\n")

  model.summary()

  print("New model 1")
  
  new_model_1 = Model(model.inputs, model.layers[-3].output)
  new_model_1.set_weights(model.get_weights())
  new_model_1.summary()

  analyzer = innvestigate.create_analyzer("lrp.z", new_model_1)
  analysis = analyzer.analyze(x_train)
  print("analysis: "+str(analysis))
  np.save("out_8_lrp", analysis)



  print("New model 2")
  
  new_model_1 = Model(model.inputs, model.layers[-3].output)
  new_model_1.set_weights(model.get_weights())
  new_model_1.summary()

  analyzer = innvestigate.create_analyzer("lrp.z", new_model_1)
  analysis = analyzer.analyze(x_train)
  print("analysis: "+str(analysis))
  np.save("out_7_lrp", analysis)

  print("New model 3")
  
  new_model_1 = Model(model.inputs, model.layers[-6].output)
  new_model_1.set_weights(model.get_weights())
  new_model_1.summary()

  analyzer = innvestigate.create_analyzer("lrp.z", new_model_1)
  analysis = analyzer.analyze(x_train)
  print("analysis: "+str(analysis))
  np.save("out_6_lrp", analysis)

  print("New model 4")
  
  new_model_1 = Model(model.inputs, model.layers[-9].output)
  new_model_1.set_weights(model.get_weights())
  new_model_1.summary()

  analyzer = innvestigate.create_analyzer("lrp.z", new_model_1)
  analysis = analyzer.analyze(x_train)
  print("analysis: "+str(analysis))
  np.save("out_5_lrp", analysis)

  print("New model 5")
  
  new_model_1 = Model(model.inputs, model.layers[-15].output)
  new_model_1.set_weights(model.get_weights())
  new_model_1.summary()

  analyzer = innvestigate.create_analyzer("lrp.z", new_model_1)
  analysis = analyzer.analyze(x_train)
  print("analysis: "+str(analysis))
  np.save("out_4_lrp", analysis)

  print("New model 6")
  
  new_model_1 = Model(model.inputs, model.layers[-18].output)
  new_model_1.set_weights(model.get_weights())
  new_model_1.summary()

  analyzer = innvestigate.create_analyzer("lrp.z", new_model_1)
  analysis = analyzer.analyze(x_train)
  print("analysis: "+str(analysis))
  np.save("out_3_lrp", analysis)

  
  print("New model 7")
  
  new_model_1 = Model(model.inputs, model.layers[-21].output)
  new_model_1.set_weights(model.get_weights())
  new_model_1.summary()

  analyzer = innvestigate.create_analyzer("lrp.z", new_model_1)
  analysis = analyzer.analyze(x_train)
  print("analysis: "+str(analysis))

  np.save("out_2_lrp", analysis)

  print("New model 8")
  
  new_model_1 = Model(model.inputs, model.layers[-24].output)
  new_model_1.set_weights(model.get_weights())
  new_model_1.summary()

  analyzer = innvestigate.create_analyzer("lrp.z", new_model_1)
  analysis = analyzer.analyze(x_train)
  print("analysis: "+str(analysis))

  np.save("out_1_lrp", analysis)

  

  
    

  # pu.db


