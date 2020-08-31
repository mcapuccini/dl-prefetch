# Imports
import click
import numpy as np

@click.command()
@click.option('--dataset-dir', required=True)
def preproc_ord_norm(dataset_dir):
  # Load data
  data = np.load(f'{dataset_dir}/deltas_split.npz')
  train = data['train']
  dev = data['dev']
  test = data['test']

  # Ordinal encoding
  train_unique, train_ord = np.unique(train, return_inverse=True)
  dev_ord = np.abs(dev.reshape(-1, 1) - train_unique).argmin(axis=1)
  test_ord = np.abs(test.reshape(-1, 1) - train_unique).argmin(axis=1)

  # Normalization
  train_norm = train_ord / (len(train_unique) - 1)
  dev_norm = dev_ord / (len(train_unique) - 1)
  test_norm = test_ord / (len(train_unique) - 1)

  # Store
  np.savez_compressed(
    f'{dataset_dir}/deltas_ord_norm.npz',
    train_norm=train_norm.astype(np.float32),
    dev_norm=dev_norm.astype(np.float32),
    test_norm=test_norm.astype(np.float32),
    train_unique=train_unique,
  )

if __name__ == '__main__':
  preproc_ord_norm() # pylint: disable=no-value-for-parameter
