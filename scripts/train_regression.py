# Imports
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
import click

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
@click.option('--dataset-dir', required=True)
def train_regression(dataset_dir, n_epochs, batch_size):
  # Load data
  train = torch.load(f'{dataset_dir}/deltas_ord_norm.train.64.pt')
  dev = torch.load(f'{dataset_dir}/deltas_ord_norm.dev.64.pt')
  max_class = len(np.load(f'{dataset_dir}/deltas_ord_norm.npz')['train_unique'])

  # Define metrics
  def accuracy(y_true_norm, y_pred_norm):
    # Denormalize output
    y_pred = torch.round(y_pred_norm * max_class).long()
    y_true = torch.round(y_true_norm * max_class).long()
    # Return accuracy
    return accuracy_score(y_true, y_pred)

  # Train
  model = Model()
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