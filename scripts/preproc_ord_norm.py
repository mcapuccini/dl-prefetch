# Imports
import click
import numpy as np
from tqdm import tqdm

def encode(to_encode, unique_to_index, train_ord):
  to_ret = np.zeros(len(to_encode), dtype=np.int64)
  p_bar = tqdm(np.ndenumerate(to_encode), desc='Encoding', total=len(to_encode))
  for idx, obj in p_bar:
    try:
      to_ret[idx] = train_ord[unique_to_index[obj]]
    except KeyError:
      to_ret[idx] = len(unique_to_index)
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
  train_unique, train_index, train_ord = np.unique(train, return_index=True, return_inverse=True)
  unique_to_index = dict(np.stack((train_unique, train_index)).T)
  dev_ord = encode(dev, unique_to_index, train_ord)
  test_ord = encode(test, unique_to_index, train_ord)

  # Normalization
  train_norm = train_ord / len(train_unique)
  dev_norm = dev_ord / len(train_unique)
  test_norm = test_ord / len(train_unique)

  # Store
  np.savez_compressed(
    f'{dataset_dir}/deltas_ord_norm.npz',
    train_ord=train_ord,
    dev_ord=dev_ord,
    test_ord=test_ord,
    train_norm=train_norm.astype(np.float32),
    dev_norm=dev_norm.astype(np.float32),
    test_norm=test_norm.astype(np.float32),
    train_unique=train_unique,
  )

if __name__ == '__main__':
  preproc_ord_norm() # pylint: disable=no-value-for-parameter
