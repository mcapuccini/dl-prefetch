import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import torch
from tqdm import tqdm

def train_loop(
  train_x,
  train_y,
  dev_x,
  dev_y,
  model,
  criterion,
  optimizer,
  n_epochs,
  batch_size,
  metrics={},
  train_y_denorm=None,
  dev_y_denorm=None,
):
  # Set warnings
  warnings.filterwarnings('ignore', category=UserWarning)

  # Figure out num batches
  n_batches = int(np.ceil(len(train_x) / batch_size))

  # History
  tr_history = {'loss': np.zeros(n_epochs * n_batches)}
  for k in metrics:
    tr_history[k] = np.zeros(n_epochs * n_batches)

  dev_history = {'loss': np.zeros(n_epochs)}
  for k in metrics:
    dev_history[k] = np.zeros(n_epochs)

  # Training loop
  for e in range(n_epochs): # epoch loop
    # Epoch progress bar
    batch_pbar = tqdm(range(n_batches), desc=f'Epoch {e+1}/{n_epochs}', file=sys.stdout)

    for b in batch_pbar: # batch loop
      # Retrieve current batch
      batch_x = train_x[b * batch_size:(b + 1) * batch_size]
      batch_y = train_y[b * batch_size:(b + 1) * batch_size]
      batch_y_denorm = train_y_denorm[b * batch_size:(b + 1) * batch_size]

      # Train on batch
      optimizer.zero_grad()
      outputs = model(batch_x)
      loss = criterion(outputs, batch_y)
      loss.backward()
      optimizer.step()
      
      # Compute batch metrics
      tr_history['loss'][e * n_batches + b] = loss.item()
      with torch.no_grad():
        for k, metric_f in metrics.items():
          tr_history[k][e * n_batches + b] = metric_f(batch_y, outputs, batch_y_denorm)

      # Updated progress bar
      p_bar_str = f'[TRAIN] Epoch: {e+1}/{n_epochs}'
      for k in tr_history:
        run_avg = np.mean(tr_history[k][:(e * n_batches + b + 1)][-100:])
        p_bar_str += f', {k}: {run_avg:.4f}'
      batch_pbar.set_description(p_bar_str)

    # Compute metrics for dev set
    with torch.no_grad():
      t0 = datetime.now() # for duration

      # Compute dev metrics
      dev_out = model(dev_x)
      dev_history['loss'][e] = criterion(dev_out, dev_y).item()
      dev_str = f"[DEVEL] Epoch: {e+1}/{n_epochs}, loss: {dev_history['loss'][e]:.4f}"
      for k, metric_f in metrics.items():
        dev_history[k][e] = metric_f(dev_y, dev_out, dev_y_denorm)
        dev_str += f', {k}: {dev_history[k][e]:.4f}'
      
      # Eval duration
      t_delta = datetime.now() - t0
      eval_duration = t_delta - timedelta(microseconds=t_delta.microseconds) # remove microseconds
      dev_str += f' (eval duration {eval_duration})'

      # Print dev metrics
      print(dev_str)
  
  # Return
  return model, tr_history, dev_history
