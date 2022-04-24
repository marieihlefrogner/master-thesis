import numpy as np
import pandas as pd
import seaborn as sns

import os
import sys

from sklearn.model_selection import train_test_split
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
STEP = step
EPOCHS = epochs
BATCH_SIZE = batch_size
filename_prefix = 'Conv1D-MADRS-pred'

def timestamp():
    return datetime.datetime.now().strftime("%m-%d-%YT%H:%M:%S")

def create_model(segment_length, optimizer, learning_rate, input_shape):
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

def create_segments_and_labels(segment_length, step):
    scores = pd.read_csv(os.path.join(DATASET_DIR, 'scores.csv'))
    scores['madrs2'].fillna(0, inplace=True)

    segments = []
    labels = []

    for person in scores['number']:
        p = scores[scores['number'] == person]
        filepath = os.path.join(DATASET_DIR, person.split('_')[0], f'{person}.csv')
        df_activity = pd.read_csv(filepath)

        for i in range(0, len(df_activity) - segment_length, step):
            segment = df_activity['activity'].values[i : i + segment_length]
            
            segments.append([segment])
            labels.append(p['madrs2'].values[0])

    segments = np.asarray(segments)
    segments = segments.reshape(-1, segment_length, 1)

    input_shape = segments.shape[1]
    segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
    labels = np.asarray(labels).astype('float32')

    return segments, labels, input_shape


img_path = f'../img/{filename_prefix}_{timestamp()}'
model_path = f'../models/{filename_prefix}_{timestamp()}'

os.mkdir(img_path)
os.mkdir(model_path)

histories = []
loss_list = []

#hours_list = [16, 24, 48, 72, 96]
hours_list = [48]

optimizer_list = ['adam', 'sgd', 'nadam']

for hours in hours_list:
    seg = hours * 60

    print(f'Segment length: {seg} ({hours} hours)')

    segments, labels, input_shape = create_segments_and_labels(seg, STEP)
    X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

    for opt in optimizer_list:
        model = create_model(seg, opt, learning_rate, input_shape)
        
        h = model.fit(X_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=1)

        model.save(f'{model_path}/{seg}_{STEP}_{EPOCHS}_{BATCH_SIZE}_{opt}.h5')

        loss = model.evaluate(X_test, y_test)[0]

        histories.append(pd.DataFrame(h.history, index=h.epoch))
        loss_list.append(loss)

historydf = pd.concat(histories, axis=1)

metrics_reported = histories[0].columns
historydf.columns = pd.MultiIndex.from_product([optimizer_list, metrics_reported], names=['optimizer', 'metric'])

plt.clf()
historydf.xs('loss', axis=1, level='metric').plot()
plt.title('Loss')
plt.xlabel('Epochs')
plt.savefig(f'{img_path}/{timestamp()}_plot_loss_train.pdf')

plt.clf()
plt.plot(optimizer_list, loss_list)
plt.xlabel('Optimizer')
plt.ylabel('Loss')
plt.savefig(f'{img_path}/{timestamp()}_plot_loss_eval.pdf')