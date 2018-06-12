from __future__ import print_function
import keras
from keras.models import Sequential, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import numpy as np
import pandas as pd
import sys
import os

BATCH_SIZE = 8192
EPOCHS = 100000
LEARNING_RATE = 0.0001

model = Sequential()
model.add(Dense(64, input_dim=38))
model.add(Activation('selu'))


model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
# model.add(Dropout(0.4))


model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
# model.add(Dropout(0.4))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
# model.add(Dropout(0.4))


model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('selu'))
# model.add(Dropout(0.4))


model.add(Dense(8))
model.add(BatchNormalization())
model.add(Activation('selu'))
# model.add(Dropout(0.4))

model.add(Dense(1))
model.add(Activation('sigmoid'))
print(model.summary())

print("Would you like to restore a previously saved model? (y/n)")
choice = raw_input()

if (choice=='y' or choice=='Y'):
	# path = raw_input("Enter path: ")
	model = load_model('checkpoints/model.h5')

print("\n")


train_flag = True

print("Train? (y for train, n for test)")
choice = raw_input()
if (choice =='n' or choice=='N'):
	df = pd.read_csv("data/out-test.csv")
	BATCH_SIZE = df.shape[0]
	EPOCHS = 1
	train_flag = False
	
else:
	df = pd.read_csv("data/out-train.csv")


x_train = df.loc[:,['Age', 'Agency_Text1', 'Agency_Text2',
   'Agency_Text3', 'Agency_Text4', 'Sex_Code_Text1', 'Sex_Code_Text2',
   'Ethnic_Code_Text1', 'Ethnic_Code_Text2', 'Ethnic_Code_Text3',
   'Ethnic_Code_Text4', 'Ethnic_Code_Text5', 'Ethnic_Code_Text6',
   'Ethnic_Code_Text7', 'Ethnic_Code_Text8', 'Ethnic_Code_Text9',
   'Language1', 'Language2', 'LegalStatus1', 'LegalStatus2',
   'LegalStatus3', 'LegalStatus4', 'LegalStatus5', 'LegalStatus6',
   'LegalStatus7', 'CustodyStatus1', 'CustodyStatus2',
   'CustodyStatus3', 'CustodyStatus4', 'CustodyStatus5',
   'CustodyStatus6', 'MaritalStatus1', 'MaritalStatus2',
   'MaritalStatus3', 'MaritalStatus4', 'MaritalStatus5',
   'MaritalStatus6', 'MaritalStatus7']].values

y_train = df["DecileScore"].values
# y_train = keras.utils.np_utils.to_categorical(y_train)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(monitor='acc', filepath="checkpoints/model.h5", verbose=True,
                                   save_best_only = True)
earlystopping = EarlyStopping(monitor='acc', min_delta=1e-6, patience=20, verbose=True)

if train_flag:
	model.fit(x_train, y_train,
	          epochs=EPOCHS,
	          batch_size=BATCH_SIZE, callbacks=[checkpointer, earlystopping])
else:
	score = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE)
	print(model.metrics_names)
	print(score)

sys.exit(0)