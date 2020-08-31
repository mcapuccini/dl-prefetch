# Imports
import click
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm

@click.command()
@click.option('--dataset-dir', required=True)
@click.option('--n-jobs', default=-1, type=int)
@click.option('--backend', default='threading')
def preproc_ord_norm(dataset_dir, n_jobs, backend):
  # Load data
  data = np.load(f'{dataset_dir}/deltas_split.npz')
  train = data['train']
  dev = data['dev']
  test = data['test']

  # Ordinal encoding
  train_unique, train_ord = np.unique(train, return_inverse=True)

  def encode(e):
    return np.abs(e - train_unique).argmin()

  with parallel_backend(backend, n_jobs=n_jobs):
    dev_ord = np.array(Parallel()(delayed(encode)(e) for e in tqdm(dev, desc='Dev encoding')))
    test_ord = np.array(Parallel()(delayed(encode)(e) for e in tqdm(test, desc='Test encoding')))

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
