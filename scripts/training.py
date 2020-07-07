# %% Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from torch import nn, optim

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
print('Number of unique deltas:', len(tr_unique))
print('Number of unique filtered deltas:', len(tr_unique_filtered))

# %% Limit out deltas to 50K
tr_most_common = tr_stacked[np.argsort(tr_stacked[:, 1])][-50000:][:, 0]
num_out_deltas = len(tr_most_common)
print('Number of output deltas:', num_out_deltas)

# %% Prepare ordinal encoders
dummy_delta = tr_unique_filtered.max() + 1
feature_enc = OrdinalEncoder()
labels_enc = OrdinalEncoder()
feature_enc.fit(np.append(tr_unique_filtered, dummy_delta).reshape(-1, 1))
labels_enc.fit(np.append(tr_most_common, dummy_delta).reshape(-1, 1))

# %% Windowing and encoding data
def window_enc(series, look_back):
  # Windowing the series
  to_ret = []
  for t in range(len(series) - look_back):
      to_ret.append(series[t:t+look_back+1].tolist())
  wdata = np.array(to_ret)
  # Get features and labels
  features = wdata[:, :look_back].reshape(-1, 1)
  labels = wdata[:, look_back].reshape(-1, 1)
  # Set dummy deltas
  features[~np.in1d(features, tr_unique_filtered)] = dummy_delta # replace rare deltas from features
  labels[~np.in1d(labels, tr_most_common)] = dummy_delta # replace deltas not included in out vocab
  # Ordinal encoding
  features = feature_enc.transform(features)
  labels = labels_enc.transform(labels)
  return features.reshape(-1, look_back), to_categorical(labels.reshape(1, -1)[0], num_classes=num_out_deltas+1)

look_back = 3
train_x, train_y = window_enc(train_deltas, look_back)
dev_x, dev_y = window_enc(dev_deltas, look_back)

# %% Define model architecture
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.embedding = nn.Embedding(
      num_embeddings=int(train_x.max()) + 1,
      embedding_dim=10,
    )
    self.lstm = nn.LSTM(
        input_size=look_back,
        hidden_size=50,
        batch_first=True)
    self.dropout = nn.Dropout(p=0.1)
    self.linear = nn.Linear(
      in_features=50,
      out_features=num_out_deltas+1
    )
  def forward(self, x):
    out = self.embedding(x)
    out, _ = self.lstm(out) # h0, c0 default to 0
    out = self.dropout(out[:, -1, :]) # [:, -1, :] slices out the last hidden state
    out = self.linear(out)
    return out
  
# %% Train
epochs=20
criterion = nn.CrossEntropyLoss()
model = Model()
optimizer = optim.Adam(model.parameters())
