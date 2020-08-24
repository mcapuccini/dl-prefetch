# Imports
from math import ceil

import click
import numpy as np
import pandas as pd

@click.command()
@click.option('--dataset-dir', required=True)
@click.option('--heldout-fract', default=0.01, type=float)
def preproc_split(dataset_dir, heldout_fract):
  # Load data
  data = pd.read_feather(f'{dataset_dir}/trace_with_deltas.feather')['delta'].to_numpy()

  # Split
  n_heldout = ceil(len(data) * heldout_fract)
  test = data[-n_heldout:]
  train_dev = data[:-n_heldout]
  train = train_dev[:-n_heldout]
  dev = train_dev[-n_heldout:]
  assert (len(train) + len(test) + len(dev) == len(data))
  assert (len(test) == len(dev))

  # Save
  np.savez_compressed(
    f'{dataset_dir}/deltas_train_dev.npz',
    train=train,
    dev=dev
  )
  np.save(f'{dataset_dir}/deltas_test.npy', test)

if __name__ == '__main__':
  preproc_split() # pylint: disable=no-value-for-parameter
