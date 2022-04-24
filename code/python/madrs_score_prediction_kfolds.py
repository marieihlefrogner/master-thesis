import numpy as np
import pandas as pd
import seaborn as sns

import os
import sys

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Nadam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from parse_args import *

DATASET_DIR = '../datasets'
SEGMENT_LENGTH = segment_length
STEP = step
EPOCHS = epochs
BATCH_SIZE = batch_size

""" Create segments and labels """

scores = pd.read_csv(os.path.join(DATASET_DIR, 'scores.csv'))
scores['madrs2'].fillna(0, inplace=True)

segments = []
labels = []

for person in scores['number']:
    p = scores[scores['number'] == person]
    filepath = os.path.join(DATASET_DIR, person.split('_')[0], f'{person}.csv') # ../datasets/[control or condition]/[person].csv
    df_activity = pd.read_csv(filepath)

    for i in range(0, len(df_activity) - SEGMENT_LENGTH, STEP):
        segment = df_activity['activity'].values[i : i + SEGMENT_LENGTH]
        
        segments.append([segment])
        labels.append(p['madrs2'].values[0])

segments = np.asarray(segments)
segments = segments.reshape(-1, SEGMENT_LENGTH, 1)

input_shape = segments.shape[1]
segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
labels = np.asarray(labels).astype('float32')

""" Create model """

def create_model(optimizer='adam', learning_rate=0.05, model_path=None, segment_length=None, input_shape=None):
    K.clear_session()

    if model_path:
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(Reshape((segment_length, 1), input_shape=(input_shape,)))
        model.add(Conv1D(128, 2, activation='relu', input_shape=(segment_length, 1)))
        model.add(MaxPooling1D(pool_size=2, strides=1))
        model.add(Conv1D(64, 2, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Flatten())
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='linear'))

        if optimizer == 'sgd':
            if learning_rate:
                optimizer = SGD(lr=learning_rate, nesterov=True)
            else:
                optimizer = SGD(nesterov=True)
        elif optimizer == 'nadam':
            if learning_rate:
                optimizer = Nadam(lr=learning_rate)

        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

    return model

""" Train model """

segments_train, segments_test, labels_train, labels_test = train_test_split(segments, labels, test_size=0.2)

skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
splits = skf.split(segments_train, labels_train)

fold_i = 0
results = []

for train_indexes, val_indexes in splits:
    print(f'Fold: {fold_i+1}/{k_folds}')

    X_train, X_val = segments_train[train_indexes], segments_train[val_indexes]
    y_train, y_val = labels_train[train_indexes], labels_train[val_indexes]

    model = create_model(optimizer=optimizer, learning_rate=learning_rate, model_path=model_path, segment_length=SEGMENT_LENGTH, input_shape=input_shape)

    if not model_path:
        h = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[], validation_data=(X_val, y_val), verbose=1)

        model.save(f'../models/madrs_score_{identifier}.h5')
        
        evaluation = model.evaluate(segments_test, labels_test)
        print(evaluation)
        results.append(evaluation[0])

    else:
        results.append(model.evaluate(X_val, y_val)[0])

    fold_i += 1

df = pd.DataFrame(results, columns=['mse'])
df.to_csv(f'../logs/result_3-folds_madrs-pred_{identifier}.txt')
