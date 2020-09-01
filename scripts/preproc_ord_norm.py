# Imports
import shutil
from os import path
from tempfile import mkdtemp

import click
import numpy as np
from joblib import Parallel, delayed, dump, load, parallel_backend
from tqdm import tqdm

def encode_obj(idx, obj, unique, out):
  encoded = np.abs(obj - unique).argmin()
  out[idx] = encoded

def encode(to_encode, unique, n_jobs, backend):
  # Create mmap for dumping output
  tmp_folder = mkdtemp()
  out_mmap_f = path.join(tmp_folder, 'encoding.mmap')
  out_mmap = np.memmap(out_mmap_f, dtype=np.int64, shape=len(to_encode), mode='w+')

  # Dump unique to mmap
  unique_mmap_f = path.join(tmp_folder, 'unique.mmap')
  dump(unique, unique_mmap_f)
  unique_mmap = load(unique_mmap_f, mmap_mode='r')

  # Encode in parallel
  p_bar = tqdm(np.ndenumerate(to_encode), desc='Encoding', total=len(to_encode))
  with parallel_backend(backend, n_jobs=n_jobs):
    Parallel()(delayed(encode_obj)(idx, obj, unique_mmap, out_mmap) for idx, obj in p_bar)

  return out_mmap, tmp_folder

@click.command()
@click.option('--dataset-dir', required=True)
@click.option('--n-jobs', default=1, type=int)
@click.option('--backend', default='loky')
def preproc_ord_norm(dataset_dir, n_jobs, backend):
  # Load data
  data = np.load(f'{dataset_dir}/deltas_split.npz')
  train = data['train']
  dev = data['dev']
  test = data['test']

  # Ordinal encoding
  train_unique, train_ord = np.unique(train, return_inverse=True)
  dev_ord, tmpf_dev = encode(dev, train_unique, n_jobs, backend)
  test_ord, tmpf_test = encode(test, train_unique, n_jobs, backend)

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

  # Cleanup
  shutil.rmtree(tmpf_dev)
  shutil.rmtree(tmpf_test)

if __name__ == '__main__':
  preproc_ord_norm() # pylint: disable=no-value-for-parameter
