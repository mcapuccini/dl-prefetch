# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# %% Load data
trace_df = pd.read_feather('/traces/canneal/train.feather')
train_df = trace_df[:200000]
dev_df = trace_df[200000:][:200000]

# %% Compute deltas
train_addr = train_df['addr_int'].to_numpy()
train_deltas = train_addr[1:] - train_addr[:-1]
dev_addr = dev_df['addr_int'].to_numpy()
dev_deltas = dev_addr[1:] - dev_addr[:-1]

# %% Get unique deltas that appear at least 10 times
tr_unique, tr_count = np.unique(train_deltas, return_counts=True)
tr_stacked = np.column_stack((tr_unique, tr_count))
tr_unique_filtered = tr_stacked[tr_stacked[:, 1] >= 10][:, 0]

# %% Set rare elements as max delta + 1
dummy_delta = tr_unique_filtered.max() + 1
train_deltas[~np.in1d(train_deltas, tr_unique_filtered)] = dummy_delta
dev_deltas[~np.in1d(dev_deltas, tr_unique_filtered)] = dummy_delta

# %% Limit out deltas to 50K
tr_most_common = tr_stacked[np.argsort(tr_stacked[:, 1])][-50000:][:, 0]
num_out_deltas = len(tr_most_common)
print('Number of output deltas:', num_out_deltas)

# %% Prepare ordinal encoders
feature_enc = OrdinalEncoder()
labels_enc = OrdinalEncoder()
feature_enc.fit(train_deltas.reshape(-1, 1))
labels_enc.fit(np.append(tr_most_common, dummy_delta).reshape(-1, 1))

# %% Windowing and encoding data
def window_enc(series, look_back):
    # Windowing the series
    to_ret = []
    for t in range(len(series) - look_back):
        to_ret.append(series[t:t+look_back+1].tolist())
    wdata = np.array(to_ret)
    # Get features and labels
    features = wdata[:, :look_back]
    labels = wdata[:, look_back]
    # Replace unknown labels with dummy delta
    labels[~np.in1d(labels, tr_most_common)] = dummy_delta
    # Ordinal encoding
    features = feature_enc.transform(features.reshape(-1, 1))
    labels = labels_enc.transform(labels.reshape(-1, 1))
    return features.reshape(-1, look_back), to_categorical(labels.reshape(1, -1)[0])

look_back = 3
train_x, train_y = window_enc(train_deltas, look_back)
dev_x, dev_y = window_enc(dev_deltas, look_back)

# %% Define model architecture
i = Input(shape=(look_back,))
x = Embedding(int(train_x.max()) + 1, 10, input_length=look_back)(i)
x = LSTM(50)(x)
x = Dropout(0.1)(x)
x = Dense(num_out_deltas+1, activation='softmax')(x)

# %% Compile model
model = Model(i, x)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# %% Train
r = model.fit(
    train_x, train_y, 
    validation_data=(dev_x, dev_y),
    epochs=20,
    shuffle=False,
    batch_size=256
)

# %% Plot loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# %% Plot accuracy
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
