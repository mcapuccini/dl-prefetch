# %% Imports
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder
from torch import nn, optim
from tqdm import tqdm

# %% Load data
trace_df = pd.read_feather('/traces/canneal/train.feather')
train_df = trace_df[100000:][:200000]
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
  # Return torch tensors
  torch_features = torch.from_numpy(features.reshape(-1, look_back).astype(np.int64))
  torch_labels = torch.from_numpy(labels.reshape(1, -1)[0].astype(np.int64))
  return torch_features, torch_labels

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
        input_size=10,
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

model = Model()

# %% Set loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# %% Train
# Params
n_epochs=20
batch_size=256
n_batches = int(np.ceil(len(train_x) / batch_size))

# History
tr_loss_history = np.zeros(n_epochs * n_batches)
tr_acc_history = np.zeros(n_epochs * n_batches)
dev_loss_history = np.zeros(n_epochs)
dev_acc_history = np.zeros(n_epochs)

# Training loop
for e in range(n_epochs): # epoch loop
  batch_pbar = tqdm(range(n_batches), desc=f'Epoch {e+1}/{n_epochs}', file=sys.stdout)
  for b in batch_pbar: # train on batch loop
    # Retrieve current batch
    batch_x = train_x[b * batch_size:(b + 1) * batch_size]
    batch_y = train_y[b * batch_size:(b + 1) * batch_size]
    # Train on batch
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    tr_loss_history[e * n_batches + b] = loss.item() # update history
    # Compute batch accuracy
    _, pred = torch.max(outputs.data, 1)
    tr_acc_history[e * n_batches + b] = (pred == batch_y).sum().item() / batch_y.size(0)
    # Updated progress bar
    loss_run_avg = np.mean(tr_loss_history[:(e * n_batches + b + 1)][-100:]) # run avg
    acc_run_avg = np.mean(tr_acc_history[:(e * n_batches + b + 1)][-100:]) # run avg
    batch_pbar.set_description(
      f'Epoch {e+1}/{n_epochs} (loss {loss_run_avg:.4f}, acc {acc_run_avg:.4f})')
  # Compute metrics for dev set
  with torch.no_grad():
    t0 = datetime.now() # for duration
    # Compute accturacy
    dev_out = model(dev_x)
    _, dev_pred = torch.max(dev_out.data, 1)
    dev_acc_history[e] = (dev_pred == dev_y).sum().item() / dev_y.size(0)
    # Compute loss
    dev_loss_history[e] = criterion(dev_out, dev_y).item()
    # Print with duration
    t_delta = datetime.now() - t0
    eval_duration = t_delta - timedelta(microseconds=t_delta.microseconds)
    print(
      f'Epoch {e+1}/{n_epochs}: dev loss {dev_loss_history[e]:.4f}, dev acc {dev_acc_history[e]:.4f} (eval duration {eval_duration})')
    
# %% Save model and history to disk
os.mkdir('/traces/canneal/200k')
torch.save(model.state_dict(), '/traces/canneal/200k/model.pth')
np.savez('/traces/canneal/200k/history.npz',
  tr_loss_history = tr_loss_history,
  tr_acc_history = tr_acc_history,
  dev_loss_history = dev_loss_history,
  dev_acc_history = dev_loss_history)
