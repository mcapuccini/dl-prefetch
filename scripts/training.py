# Imports
import os
import sys
import uuid
import warnings
from datetime import datetime, timedelta

import click
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report)
from sklearn.preprocessing import OrdinalEncoder
from torch import nn, optim
from tqdm import tqdm

# Define model architecture
class Model(nn.Module):
  def __init__(
    self,
    num_embeddings,
    embedding_dim,
    lstm_hidden_size,
    dropout_p,
    linear_out_features,
  ):
    super(Model, self).__init__()
    self.embedding = nn.Embedding(
      num_embeddings=num_embeddings,
      embedding_dim=embedding_dim,
    )
    self.lstm = nn.LSTM(
      input_size=embedding_dim,
      hidden_size=lstm_hidden_size,
      batch_first=True,
    )
    self.dropout = nn.Dropout(p=dropout_p)
    self.linear = nn.Linear(
      in_features=lstm_hidden_size,
      out_features=linear_out_features,
    )

  def forward(self, x):
    out = self.embedding(x)
    out, _ = self.lstm(out) # h0, c0 default to 0
    out = self.dropout(out[:, -1, :]) # [:, -1, :] gets the last state
    out = self.linear(out)
    return out

@click.command('training')
@click.option('--train-set-path', required=True)
@click.option('--out-dir', required=True)
@click.option('--train-set-size', default=None, type=int)
@click.option('--dev-set-size', default=500000, type=int)
@click.option('--skip-size', default=0, type=int)
@click.option('--occurrence-thr', default=10, type=int)
@click.option('--max-out-deltas', default=50000, type=int)
@click.option('--look-back', default=3, type=int)
@click.option('--embedding-dim', default=10, type=int)
@click.option('--lstm-hidden-size', default=50, type=int)
@click.option('--dropout-p', default=0.1, type=float)
@click.option('--n-epochs', default=20, type=int)
@click.option('--batch-size', default=256, type=int)
def training(
  train_set_path,
  out_dir,
  train_set_size,
  dev_set_size,
  skip_size,
  occurrence_thr,
  max_out_deltas,
  look_back,
  embedding_dim,
  lstm_hidden_size,
  dropout_p,
  n_epochs,
  batch_size,
):
  # Make out dir
  full_out_dir = f'{out_dir}/{uuid.uuid4().hex}'
  os.makedirs(full_out_dir)

  # Print out and save parameters
  parameters = {}
  parameters['train_set_path'] = train_set_path
  parameters['out_dir'] = full_out_dir
  parameters['train_set_size'] = train_set_size
  parameters['dev_set_size'] = dev_set_size
  parameters['skip_size'] = skip_size
  parameters['occurrence_thr'] = occurrence_thr
  parameters['max_out_deltas'] = max_out_deltas
  parameters['look_back'] = look_back
  parameters['embedding_dim'] = embedding_dim
  parameters['lstm_hidden_size'] = lstm_hidden_size
  parameters['dropout_p'] = dropout_p
  parameters['n_epochs'] = n_epochs
  parameters['batch_size'] = batch_size
  parameters_df = pd.DataFrame(parameters, index=['Parameters']).transpose()
  parameters_df.to_csv(f'{full_out_dir}/parameters.csv')
  print(parameters_df)

  # Load and split data
  trace_df = pd.read_feather(train_set_path)
  if (train_set_size is None): # arxiv-like
    train_df = trace_df[:-dev_set_size]
    dev_df = trace_df[-dev_set_size:]
  else: # memsys-like
    train_df = trace_df[skip_size:][:train_set_size]
    dev_df = trace_df[(skip_size + train_set_size):][:dev_set_size]

  # Compute deltas
  train_addr = train_df['addr_int'].to_numpy()
  train_deltas = train_addr[1:] - train_addr[:-1]
  dev_addr = dev_df['addr_int'].to_numpy()
  dev_deltas = dev_addr[1:] - dev_addr[:-1]

  # Filter rare deltas
  tr_unique, tr_count = np.unique(train_deltas, return_counts=True)
  tr_stacked = np.column_stack((tr_unique, tr_count))
  tr_unique_filtered = tr_stacked[tr_stacked[:, 1] >= occurrence_thr][:, 0]
  print('Number of unique deltas:', len(tr_unique))
  print('Number of unique filtered deltas:', len(tr_unique_filtered))

  # Get output deltas
  tr_most_common = tr_stacked[np.argsort(tr_stacked[:, 1])][-max_out_deltas:][:, 0]
  num_out_deltas = len(tr_most_common)
  print('Number of output deltas:', num_out_deltas)

  # Prepare ordinal encoders
  dummy_delta = tr_unique_filtered.max() + 1
  feature_enc = OrdinalEncoder()
  labels_enc = OrdinalEncoder()
  feature_enc.fit(np.append(tr_unique_filtered, dummy_delta).reshape(-1, 1))
  labels_enc.fit(np.append(tr_most_common, dummy_delta).reshape(-1, 1))
  joblib.dump(feature_enc, f'{full_out_dir}/feature_enc.joblib.gz', compress='gzip')
  joblib.dump(labels_enc, f'{full_out_dir}/labels_enc.joblib.gz', compress='gzip')

  # Encoding and widowing
  def window_enc(series):
    # Windowing the series
    to_ret = []
    pbar = tqdm(range(len(series) - look_back), desc='Data windowing', file=sys.stdout)
    for t in pbar:
      to_ret.append(series[t:t + look_back + 1].tolist())
    wdata = np.array(to_ret)
    # Get features and labels
    features = wdata[:, :look_back].reshape(-1, 1)
    labels = wdata[:, look_back].reshape(-1, 1)
    # Set dummy deltas
    dummy_mask_f = ~np.in1d(features, tr_unique_filtered)
    dummy_mask_l = ~np.in1d(labels, tr_most_common)
    features[dummy_mask_f] = dummy_delta # replace rare deltas from features
    labels[dummy_mask_l] = dummy_delta # replace deltas not included in out vocab
    # Ordinal encoding
    features = feature_enc.transform(features)
    labels = labels_enc.transform(labels)
    # Return torch tensors and dummy label fraction
    torch_features = torch.from_numpy(features.reshape(-1, look_back).astype(np.int64))
    torch_labels = torch.from_numpy(labels.reshape(1, -1)[0].astype(np.int64))
    frac_dummy_f = np.sum(dummy_mask_f) / (len(torch_features) * look_back)
    frac_dummy_l = np.sum(dummy_mask_l) / len(torch_labels)
    return torch_features, torch_labels, frac_dummy_f, frac_dummy_l

  train_x, train_y, df_tr, dl_tr = window_enc(train_deltas)
  dev_x, dev_y, df_dev, dl_dev = window_enc(dev_deltas)
  torch.save(train_x, f'{full_out_dir}/train_x.pth')
  torch.save(train_y, f'{full_out_dir}/train_y.pth')
  torch.save(dev_x, f'{full_out_dir}/dev_x.pth')
  torch.save(dev_y, f'{full_out_dir}/dev_y.pth')
  print(f'Dummy features fraction in train set: {df_tr:.4f})')
  print(f'Dummy labels fraction in train set: {dl_tr:.4f})')
  print(f'Dummy features fraction in dev set: {df_dev:.4f})')
  print(f'Dummy labels fraction in dev set: {dl_dev:.4f})')

  # Define model, loss and optimizer
  model = Model(
    num_embeddings=int(train_x.max()) + 1,
    embedding_dim=embedding_dim,
    lstm_hidden_size=lstm_hidden_size,
    dropout_p=dropout_p,
    linear_out_features=num_out_deltas + 1,
  )
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters())

  # Init history
  n_batches = int(np.ceil(len(train_x) / batch_size))
  tr_loss_h = np.zeros(n_epochs * n_batches)
  tr_acc_h = np.zeros(n_epochs * n_batches)
  tr_bal_acc_h = np.zeros(n_epochs * n_batches)
  dev_loss_h = np.zeros(n_epochs)
  dev_acc_h = np.zeros(n_epochs)
  dev_bal_acc_h = np.zeros(n_epochs)

  # Training loop
  warnings.filterwarnings('ignore', category=UserWarning)
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
      tr_loss_h[e * n_batches + b] = loss.item() # update history
      # Compute batch accuracy
      _, pred = torch.max(outputs.data[:, :-1], 1) # [:,:-1] to remove dummy class
      tr_acc_h[e * n_batches + b] = accuracy_score(batch_y, pred)
      tr_bal_acc_h[e * n_batches + b] = balanced_accuracy_score(batch_y, pred)
      # Updated progress bar
      loss_run_avg = np.mean(tr_loss_h[:(e * n_batches + b + 1)][-100:])
      acc_run_avg = np.mean(tr_acc_h[:(e * n_batches + b + 1)][-100:])
      bal_acc_run_avg = np.mean(tr_bal_acc_h[:(e * n_batches + b + 1)][-100:])
      batch_pbar.set_description(f'Epoch {e+1}/{n_epochs} '
                                 f'(loss {loss_run_avg:.4f}, '
                                 f'acc {acc_run_avg:.4f}, '
                                 f'bal acc {bal_acc_run_avg:.4f})')
    # Compute metrics for dev set
    with torch.no_grad():
      t0 = datetime.now() # for duration
      # Compute accturacy
      dev_out = model(dev_x)
      _, dev_pred = torch.max(dev_out.data[:, :-1], 1) # [:,:-1] to remove dummy class
      dev_acc_h[e] = accuracy_score(dev_y, dev_pred)
      dev_bal_acc_h[e] = balanced_accuracy_score(dev_y, dev_pred)
      # Compute loss
      dev_loss_h[e] = criterion(dev_out, dev_y).item()
      # Print with duration
      t_delta = datetime.now() - t0
      eval_duration = t_delta - timedelta(microseconds=t_delta.microseconds) # remove microseconds
      print(f'Epoch {e+1}/{n_epochs}: '
            f'dev loss {dev_loss_h[e]:.4f}, '
            f'dev acc {dev_acc_h[e]:.4f}, '
            f'dev bal acc {dev_bal_acc_h[e]:.4f} '
            f'(eval duration {eval_duration})')

  # Save model, history and last predictions
  torch.save(model, f'{full_out_dir}/model.pth')
  np.savez_compressed(
    f'{full_out_dir}/history.npz',
    tr_loss_h=tr_loss_h,
    tr_acc_h=tr_acc_h,
    tr_bal_acc_h=tr_bal_acc_h,
    dev_loss_h=dev_loss_h,
    dev_acc_h=dev_acc_h,
    dev_bal_acc_h=dev_bal_acc_h,
  )
  torch.save(dev_pred, f'{full_out_dir}/dev_pred.pth')

  # Compute/Save/Print classification report
  report = classification_report(dev_y, dev_pred, output_dict=True)
  report_df = pd.DataFrame(report).transpose().drop('accuracy')
  report_df.to_csv(f'{full_out_dir}/report_df.csv.gz', compression='gzip')
  print(report_df)

  # Compute/Save/Print metrics DF
  metrics = {}
  metrics['accuracy'] = dev_acc_h[-1]
  metrics['bal accuracy'] = dev_bal_acc_h[-1]
  metrics['num unique deltas'] = len(tr_unique)
  metrics['num unique filtered deltas'] = len(tr_unique_filtered)
  metrics['num out deltas'] = len(tr_unique_filtered)
  metrics['tr dummy feature fract'] = df_tr
  metrics['tr dummy labels fract'] = dl_tr
  metrics['dev dummy feature fract'] = df_dev
  metrics['dev dummy labels fract'] = dl_dev
  metrics_df = pd.DataFrame(metrics, index=['Metrics']).transpose()
  metrics_df.to_csv(f'{full_out_dir}/metrics_df.csv')
  print(metrics_df)

if __name__ == '__main__':
  training() # pylint: disable=no-value-for-parameter
