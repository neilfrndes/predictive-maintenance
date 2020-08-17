import logging

import keras
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn import preprocessing

# Setup logging
logger = logging.getLogger(__name__)

# Setting random seed
np.random.seed(1234)
PYTHONHASHSEED = 0
min_sequence_length = 50

# Load dataset and label columns
train_df = pd.read_csv('data/train.csv', sep=" ", header=None)
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
    gen_sequence(train_df[train_df['id'] == id], min_sequence_length,
                 sequence_cols)) for id in train_df['id'].unique())

# generate sequences and convert to numpy array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
seq_array.shape


# function to generate labels
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


# generate labels
label_gen = [
    gen_labels(train_df[train_df['id'] == id], min_sequence_length, ['label1'])
    for id in train_df['id'].unique()
]
label_array = np.concatenate(label_gen).astype(np.float32)

# Build LSTM network
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()

model.add(
    LSTM(input_shape=(min_sequence_length, nb_features),
         units=100,
         return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.4))

model.add(Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

logger.info(model.summary())

model.fit(seq_array,
          label_array,
          epochs=10,
          batch_size=200,
          validation_split=0.05,
          verbose=1,
          callbacks=[
              keras.callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=0,
                                            verbose=0,
                                            mode='auto')
          ])

model.save('model')
