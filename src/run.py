# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import logging
from timeit import default_timer as timer

import keras
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from common import STATS, calculate_stats

# Setup logging
LOG_LEVEL = logging.INFO
logging.basicConfig(format='%(levelname)s:%(message)s', level=LOG_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

logger.info("Starting benchmark...")

# Setting random seed for reproducibility
np.random.seed(1234)
PYTHONHASHSEED = 0

# Parameters
NUM_LOOPS = 10
MIN_SEQUENCE_LENGTH = 50
TRAINING_PARAMS = dict(
    epochs=10,
    batch_size=200,
    validation_split=0.05,
    verbose=1,
)
INFERENCING_PARAMS = dict(
    verbose=1,
    batch_size=200
)

# TRAINING DATA
# Load dataset and label columns
train_df = pd.read_csv('../data/train.csv', sep=" ", header=None)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = [
    'id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4',
    's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15',
    's16', 's17', 's18', 's19', 's20', 's21'
]

# Generate remaining useful life (RUL) feature
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)

# generate label columns for training data
w1 = 30
w0 = 15
train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0)
train_df['label2'] = train_df['label1']
train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

# MinMax normalization
train_df['cycle_norm'] = train_df['cycle']
cols_normalize = train_df.columns.difference(
    ['id', 'cycle', 'RUL', 'label1', 'label2'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(
    train_df[cols_normalize]),
                             columns=cols_normalize,
                             index=train_df.index)
join_df = train_df[train_df.columns.difference(cols_normalize)].join(
    norm_train_df)
train_df = join_df.reindex(columns=train_df.columns)


# function to reshape features into (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length),
                           range(seq_length, num_elements)):
        yield data_array[start:stop, :]


# pick the feature columns
sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)

# generator for the sequences
seq_gen = (list(
    gen_sequence(train_df[train_df['id'] == id], MIN_SEQUENCE_LENGTH,
                 sequence_cols)) for id in train_df['id'].unique())

# generate sequences and convert to numpy array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

# function to generate labels
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]

# generate labels
label_gen = [
    gen_labels(train_df[train_df['id'] == id], MIN_SEQUENCE_LENGTH, ['label1'])
    for id in train_df['id'].unique()
]
label_array = np.concatenate(label_gen).astype(np.float32)


# TEST DATA
# read test data
test_df = pd.read_csv('../data/test.csv', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = [
    'id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4',
    's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15',
    's16', 's17', 's18', 's19', 's20', 's21'
]

# read ground truth data
truth_df = pd.read_csv('../data/truth.csv', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

# min max normalization
test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                            columns=cols_normalize,
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(
    norm_test_df)
test_df = test_join_df.reindex(columns=test_df.columns)
test_df = test_df.reset_index(drop=True)

# generate RUL for test data
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = rul['max'] + truth_df['more']
truth_df.drop('more', axis=1, inplace=True)

test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)

# generate label columns w0 and w1 for test data
test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0)
test_df['label2'] = test_df['label1']
test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2


# Build LSTM network
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()

model.add(
    LSTM(input_shape=(MIN_SEQUENCE_LENGTH, nb_features),
         units=100,
         return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.4))

model.add(Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

logger.debug(model.summary())

# Benchmark training
train_times = []
for _ in range(NUM_LOOPS):
    start_time_train = timer()
    model.fit(seq_array,
            label_array,
            **TRAINING_PARAMS,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0,
                                                patience=0,
                                                verbose=0,
                                                mode='auto')
            ])
    end_time_train = timer()
    train_times.append(end_time_train-start_time_train)
logger.info('Training Benchmark \n%s\n%s\n\n', STATS, calculate_stats(train_times))

# Save model to inspect later
model.save('model')

# Training metrics
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
accuracy = scores[1]

# make predictions and compute confusion matrix
y_pred = model.predict_classes(seq_array, **INFERENCING_PARAMS)
y_true = label_array
cm = confusion_matrix(y_true, y_pred)

# compute precision and recall
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
logger.info(
    "\nTRAINING METRICS\n"
    "Num Records: %d"
    "Accuracy: %.2f\n"
    "Precision: %.2f\n"
    "Recall: %.2f\n"
    "Confusion Matrix: \n%s\n", len(label_array), accuracy, precision, recall, cm)


# Test metrics
seq_array_test_last = [
    test_df[test_df['id'] == id][sequence_cols].values[-MIN_SEQUENCE_LENGTH:]
    for id in test_df['id'].unique()
    if len(test_df[test_df['id'] == id]) >= MIN_SEQUENCE_LENGTH
]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
y_mask = [
    len(test_df[test_df['id'] == id]) >= MIN_SEQUENCE_LENGTH
    for id in test_df['id'].unique()
]
label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(
    label_array_test_last.shape[0], 1).astype(np.float32)
label_array_test_last.shape

scores_test = model.evaluate(seq_array_test_last,
                             label_array_test_last,
                             verbose=2)
accuracy_test = scores_test[1]

# make predictions and compute confusion matrix
# Benchmark inferencing
test_times = []
for _ in range(NUM_LOOPS):
    start_time_test = timer()
    y_pred_test = model.predict_classes(seq_array_test_last)
    end_time_test = timer()
    test_times.append(end_time_test-start_time_test)
logger.info('Inferencing Benchmark \n%s\n%s\n\n', STATS, calculate_stats(test_times))

y_true_test = label_array_test_last
cm_test = confusion_matrix(y_true_test, y_pred_test)

precision_test = precision_score(y_true_test, y_pred_test)
recall_test = recall_score(y_true_test, y_pred_test)
f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)

logger.info(
    "\nINFERENCING METRICS\n"
    "Num Records: %d"
    "Accuracy: %.2f\n"
    "Precision: %.2f\n"
    "Recall: %.2f\n"
    "F1 score: %.2f\n"
    "Confusion Matrix: \n%s", len(label_array_test_last), accuracy_test, precision_test, recall_test, f1_test, cm_test)
