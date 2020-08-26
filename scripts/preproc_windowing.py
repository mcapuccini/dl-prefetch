# Imports
import numpy as np
import torch
import click

def windowing(series, lookback):
  with torch.no_grad():
    series_torch = torch.from_numpy(series)
    windowed_series = series_torch.unfold(0, lookback, 1)
  return windowed_series


@click.command()
@click.option('--dataset-dir', required=True)
@click.option('--win-size', default=64, type=int)
def preproc_windowing(dataset_dir, win_size):
  # Load
  data = np.load(f'{dataset_dir}/deltas_ord_norm.npz')
  train = data['train_norm']
  dev = data['dev_norm']
  test = data['test_norm']

  # Windowing
  train_win = windowing(train, win_size)
  dev_win = windowing(dev, win_size)
  test_win = windowing(test, win_size)

  # Save
  torch.save(train_win, f'{dataset_dir}/deltas_ord_norm.train.{win_size}.pt')
  torch.save(dev_win, f'{dataset_dir}/deltas_ord_norm.dev.{win_size}.pt')
  torch.save(test_win, f'{dataset_dir}/deltas_ord_norm.test.{win_size}.pt')

if __name__ == '__main__':
  preproc_windowing() # pylint: disable=no-value-for-parameter