import os
import sys
import random

import numpy as np
import pandas as pd

from tensorflow.keras import backend as K

from cnn.models import ClassificationModel
from cnn.utils import create_segments_and_labels

DATASET_DIR = '../datasets'
segment_length = 480
step = 60

segments, labels, input_shape = create_segments_and_labels(DATASET_DIR, segment_length, step)
model = ClassificationModel(input_shape=input_shape, segment_length=segment_length, step=step, optimizer='adam', verbose=1, dropout=0.5, n_output_classes=2)
model.plot('hei.png')