import numpy as np
import pandas as pd
import seaborn as sns

import os
import sys
import time
import random
import datetime

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from IPython.display import display

from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from parse_args import *

pd.options.mode.chained_assignment = None

PROJECT_DIR = '..'
DATASET_DIR = f'{PROJECT_DIR}/datasets'
SAVE_DIR = f'{PROJECT_DIR}/data'

if verbose:
    verbose = 1
    print('Verbose mode.')
else:
    verbose = 0

CATEGORIES = ['Condition', 'Control']
LABELS = ['normal', 'bipolar']

MADRS_LABLES = ['Normal', 'Mild', 'Moderate', 'Severe']
MADRS_VALUES = [0, 7, 20, 34]

log = None

def setup():
    global log

    if logfile:
        log = open(logfile, 'w')

def cleanup():
    global log

    if logfile:
        log.close()

def make_confusion_matrix(validations, predictions, output_file=None, print_stdout=False, xticklabels=LABELS, yticklabels=LABELS):
    global log

    matrix = confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=xticklabels[:matrix.shape[0]],
                yticklabels=yticklabels[:matrix.shape[1]],
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    if output_file:
        plt.savefig(output_file)

    if print_stdout:
        print('Confusion matrix:\n', matrix)

    if logfile:
        log.write(
            ('=========================\n'
            'Input Data:\n'
            'Confusion Matrix:\n'
            f'{matrix}\n'
            '=========================\n\n')
        )

def average_str(string):
    if(type(string) == str):
        num1, num2 = string.split('-')
        num1 = int(num1)
        num2 = int(num2)

        return (num1 + num2) // 2

    else:
        return string

def time_between(now, start, end):
    if start <= end:
        return start <= now < end
    else:
        return start <= now or now < end

def extract_time_of_day(timestamp):
    ts = time.strptime(timestamp.split(' ')[1], '%H:%M:%S')

    for label in part_of_days.keys():
        times = part_of_days[label]
        start = time.strptime(times[0], '%H:%M:%S')
        end = time.strptime(times[1], '%H:%M:%S')

        if time_between(ts, start, end):
            return label

    return None

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma

def is_at_night(timestamp):
    return time_between(time.strptime(timestamp.split(' ')[1], '%H:%M:%S'),
                        time.strptime('22:00:00', '%H:%M:%S'),
                        time.strptime('06:00:00', '%H:%M:%S'))

def is_daytime(timestamp):
    return time_between(time.strptime(timestamp.split(' ')[1], '%H:%M:%S'),
                        time.strptime('06:00:00', '%H:%M:%S'),
                        time.strptime('21:00:00', '%H:%M:%S'))

def create_segments_and_labels_madrs(n_features, segment_length, step, k_folds=1):
    global log

    scores = pd.read_csv(os.path.join(DATASET_DIR, 'scores.csv'))
    scores['madrs2'].fillna(0, inplace=True)
    
    classes = len(MADRS_VALUES)

    segments = []
    labels = []

    for person in scores['number']:
        p = scores[scores['number'] == person]
        filepath = os.path.join(DATASET_DIR, person.split('_')[0], f'{person}.csv')
        df_activity = pd.read_csv(filepath)

        for i in range(0, len(df_activity) - segment_length, step):
            segment = df_activity['activity'].values[i : i + segment_length]
            segments.append([segment])

            madrs = p['madrs2'].values[0]

            for i in range(classes):
                if madrs >= MADRS_VALUES[classes - i - 1]:
                    labels.append(classes - i - 1)
                    break

    segments = np.asarray(segments).reshape(-1, segment_length, n_features)

    num_time_periods, num_sensors = segments.shape[1], segments.shape[2]
    input_shape = num_time_periods * num_sensors

    segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
    
    labels = np.asarray(labels).astype('float32')

    if k_folds <= 1:
        labels = to_categorical(labels, 4)

    if verbose:
        print('\nINPUT DATA\n------------')
        print(f'Segments:', segments.shape, ':: Labels:', labels.shape)
        print(f'num_time_periods: {num_time_periods}, num_sensors: {num_sensors}, input_shape: {input_shape}')
        print('------------\n')

    if logfile:
        log.write(
            ('=========================\n'
            'Input Data:\n'
            f'Segments: {segments.shape}\n'
            f'Labels: {labels.shape}\n'
            f'num_time_periods: {num_time_periods}\n'
            f'num_sensors: {num_sensors}\n'
            f'input_shape: {input_shape}\n'
            '=========================\n\n')
        )
    
    return segments, labels, num_sensors, input_shape    

def create_segments_and_labels_madrs_val(n_features, segment_length, step, k_folds=1):
    global log

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
    segments = segments.reshape(-1, segment_length, n_features)

    num_time_periods, num_sensors = segments.shape[1], segments.shape[2]
    input_shape = num_time_periods * num_sensors

    segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
    
    labels = np.asarray(labels).astype('float32')

    if k_folds <= 1:
        labels = to_categorical(labels, 2)

    if verbose:
        print('\nINPUT DATA\n------------')
        print(f'Segments:', segments.shape, ':: Labels:', labels.shape)
        print(f'num_time_periods: {num_time_periods}, num_sensors: {num_sensors}, input_shape: {input_shape}')
        print('------------\n')

    if logfile:
        log.write(
            ('=========================\n'
            'Input Data:\n'
            f'Segments: {segments.shape}\n'
            f'Labels: {labels.shape}\n'
            f'num_time_periods: {num_time_periods}\n'
            f'num_sensors: {num_sensors}\n'
            f'input_shape: {input_shape}\n'
            '=========================\n\n')
        )
    
    return segments, labels, num_sensors, input_shape

def create_segments_and_labels(n_features, segment_length, step, k_folds=1):
    global log

    scores = pd.read_csv(os.path.join(DATASET_DIR, 'scores.csv'))
    scores['afftype'].fillna(0, inplace=True)
    
    segments = []
    labels = []

    for person in scores['number']:
        p = scores[scores['number'] == person]
        filepath = os.path.join(DATASET_DIR, person.split('_')[0], f'{person}.csv')
        df_activity = pd.read_csv(filepath)

        for i in range(0, len(df_activity) - segment_length, step):
            segment = df_activity['activity'].values[i : i + segment_length]
            segments.append([segment])

            if p['afftype'].values[0] == 0:
                labels.append(0)
            else:
                labels.append(1)

    labels = np.asarray(labels).astype('float32')

    if k_folds <= 1:
        labels = to_categorical(labels, 2)
    
    segments = np.asarray(segments).reshape(-1, segment_length, n_features)

    num_time_periods, num_sensors = segments.shape[1], segments.shape[2]
    input_shape = num_time_periods * num_sensors

    segments = segments.reshape(segments.shape[0], input_shape).astype('float32')

    if verbose:
        print('\nINPUT DATA\n------------')
        print(f'Segments:', segments.shape, ':: Labels:', labels.shape)
        print(f'num_time_periods: {num_time_periods}, num_sensors: {num_sensors}, input_shape: {input_shape}')
        print('------------\n')

    if logfile:
        log.write(
            ('=========================\n'
            'Input Data:\n'
            f'Segments: {segments.shape}\n'
            f'Labels: {labels.shape}\n'
            f'num_time_periods: {num_time_periods}\n'
            f'num_sensors: {num_sensors}\n'
            f'input_shape: {input_shape}\n'
            '=========================\n\n')
        )
    
    return segments, labels, num_sensors, input_shape

def create_model(segment_length, num_sensors, input_shape, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], output_classes=2, dropout=None, verbose=1):
    global log

    K.clear_session()

    model = Sequential()
    model.add(Reshape((segment_length, num_sensors), input_shape=(input_shape,)))
    
    model.add(Conv1D(100, 10, activation='relu', input_shape=(segment_length, num_sensors)))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(GlobalAveragePooling1D())

    if dropout:
        model.add(Dropout(dropout))
    
    model.add(Dense(output_classes, activation='softmax'))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    if verbose:
        model.summary()
    
    if logfile:
        model.summary(print_fn=lambda x: log.write(x + '\n'))

    return model

def create_model_madrs(segment_length, num_sensors, input_shape, loss='mean_squared_logarithmic_error', optimizer='nadam', metrics=['mse'], dropout=0.5):
    global log

    K.clear_session()

    model = Sequential()

    model.add(Reshape((segment_length, num_sensors), input_shape=(input_shape,)))

    model.add(Conv1D(100, 10, activation='relu', input_shape=(segment_length, num_sensors)))
    model.add(Conv1D(100, 10, activation='relu'))

    model.add(MaxPooling1D(2))
    
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(Conv1D(160, 10, activation='relu'))

    model.add(GlobalAveragePooling1D())

    # model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    if verbose:
        model.summary()
    
    if logfile:
        model.summary(print_fn=lambda x: log.write(x + '\n'))

    return model

def train(model, X_train, y_train, batch_size, epochs, callbacks, validation_split=0.2, validation_data=None, verbose=1):
    if validation_data:
        return model.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=validation_data,
                    shuffle=True,
                    verbose=verbose)

    return model.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_split=validation_split,
                    verbose=verbose)

def predict(model, X_test, y_test, verbose=False):
    global log

    y_pred_test = model.predict(X_test)

    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(y_test, axis=1)

    if verbose:
        print(classification_report(max_y_test, max_y_pred_test))

    if logfile:
        log.write(
            ('=========================\n'
            'Classification Report:\n'
            f'{classification_report(max_y_test, max_y_pred_test)}\n'
            '=========================\n\n'
            )
        )

    return max_y_test, max_y_pred_test

def evaluate(model, X_test, y_test, verbose):
    global log
    
    loss, acc = model.evaluate(X_test, y_test)

    if verbose:
        print(f'Accuracy: {acc}')
        print(f'Loss: {loss}')

    if logfile:
        log.write(
            ('=========================\n'
            'Evaluation:\n'
            f'Accuracy: {acc}\n'
            f'Loss: {loss}\n'
            '=========================\n\n')
        )

    return loss, acc