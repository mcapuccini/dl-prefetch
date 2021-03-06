# Imports
import click
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim

from helpers import train_loop

# Define model architecture
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
    self.dropout = nn.Dropout(p=0.1)
    self.linear = nn.Linear(in_features=50, out_features=1)

  def forward(self, x):
    out, _ = self.lstm(x) # h0, c0 default to 0
    out = self.dropout(out[:, -1, :]) # [:, -1, :] slices out the last hidden state
    out = self.linear(out)
    return out

@click.command()
@click.option('--dataset-dir', required=True)
@click.option('--n-epochs', default=20, type=int)
@click.option('--batch-size', default=256, type=int)
@click.option('--win-size', default=64, type=int)
@click.option('--device', default='cpu')
@click.option('--num-threads', default=-1, type=int)
def train_regression(dataset_dir, n_epochs, batch_size, win_size, device, num_threads):
  # Set num threads
  if num_threads > 0:
    torch.set_num_threads(num_threads)

  # Load data
  train = torch.load(f'{dataset_dir}/deltas_ord_norm.train.{win_size}.pt').to(device)
  dev = torch.load(f'{dataset_dir}/deltas_ord_norm.dev.{win_size}.pt').to(device)
  data = np.load(f'{dataset_dir}/deltas_ord_norm.npz')
  train_y_denorm = torch.from_numpy(data['train_ord'][win_size - 1:]).to(device)
  dev_y_denorm = torch.from_numpy(data['dev_ord'][win_size - 1:]).to(device)
  max_class = len(data['train_unique'])
  assert (len(train_y_denorm) == len(train))
  assert (len(dev_y_denorm) == len(dev))

  # Metrics
  def accuracy(y_true, y_pred, y_true_denorm): # pylint: disable=unused-argument
    # Denormalize output
    y_pred_denorm = torch.round(y_pred * max_class).long()
    return accuracy_score(y_true_denorm, y_pred_denorm)

  # Train
  model = Model()
  model.to(device)
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters())
  _, tr_history, dev_history = train_loop(
    train_x=train[:, :-1].unsqueeze(2),
    train_y=train[:, -1].reshape(-1, 1),
    dev_x=dev[:, :-1].unsqueeze(2),
    dev_y=dev[:, -1].reshape(-1, 1),
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    n_epochs=n_epochs,
    batch_size=batch_size,
    train_y_denorm=train_y_denorm,
    dev_y_denorm=dev_y_denorm,
    metrics={'accuracy': accuracy},
  )

  # Save
  np.savez(
    f'{dataset_dir}/regression_history.npz',
    tr_history=tr_history,
    dev_history=dev_history,
  )

if __name__ == '__main__':
  train_regression() # pylint: disable=no-value-for-parameter
