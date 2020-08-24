# Imports
import click
import numpy as np

@click.command()
@click.option('--dataset-dir', required=True)
def preproc_ord_norm(dataset_dir):
  # Load data
  data = np.load(f'{dataset_dir}/deltas_train_dev.npz')
  train = data['train']

  # Ordinal encoding
  train_unique, train_ord = np.unique(train, return_inverse=True)

  # Normalization
  train_norm = train_ord / (len(train_unique) - 1)

  # Store
  np.savez_compressed(
    f'{dataset_dir}/deltas_train_ord_norm.npz',
    train_norm=train_norm,
    train_unique=train_unique,
  )

if __name__ == '__main__':
  preproc_ord_norm() # pylint: disable=no-value-for-parameter
