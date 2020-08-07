import pickle
from math import ceil

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


@click.command()
@click.option('--dataset-dir', required=True)
@click.option('--n-bins', default=100, type=int)
def stats(dataset_dir, n_bins):
  # Load trace and compute deltas
  trace = np.fromfile(f'{dataset_dir}/roitrace.bin', dtype=np.int64)
  deltas = trace[1:] - trace[:-1]

  # Plot
  trace_hist = plt.hist(trace, bins=100)
  deltas_hist = plt.hist(deltas, bins=100)

  # Compute unique addr/deltas
  addr_unique = np.unique(trace)
  delta_unique, delta_counts = np.unique(deltas, return_counts=True)
  rare_deltas = delta_counts[delta_counts < 10]

  # Stats
  stats = {}
  stats['trace len'] = len(trace)
  stats['unique addr'] = len(addr_unique)
  stats['unique deltas'] = len(delta_unique)
  stats['rare deltas (< 10)'] = len(rare_deltas)
  stats['unique deltas (no rare)'] = len(delta_unique) - len(rare_deltas)
  stats['deltas 50% mass'] = ceil(len(delta_unique) / 2)
  stats['deltas 50K coverage'] = 50000 / len(delta_unique)

  # Save
  pd.DataFrame(stats, index=['stats']).transpose().to_csv(f'{dataset_dir}/stats.csv')
  pickle.dump(trace_hist, open(f'{dataset_dir}/trace_hist_{n_bins}.pickle', 'wb'))
  pickle.dump(deltas_hist, open(f'{dataset_dir}/deltas_hist_{n_bins}.pickle', 'wb'))

if __name__ == '__main__':
  stats() # pylint: disable=no-value-for-parameter