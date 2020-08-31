# Imports
import click
import numpy as np

def encode(to_encode, unique):
  to_ret = np.zeros(len(to_encode))
  for i, e in np.ndenumerate(to_encode):
    to_ret[i] = np.abs(e - unique).argmin()
  return to_ret

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
  dev_ord = encode(dev, train_unique)
  test_ord = encode(test, train_unique)

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
