# Imports
import pickle
from math import ceil

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

pd.options.mode.chained_assignment = None

def stats_dict(trace, deltas):
  # Compute unique addr/deltas
  addr_unique = np.unique(trace)
  delta_unique, delta_counts = np.unique(deltas, return_counts=True)
  rare_deltas = delta_counts[delta_counts < 10]

  # Stats
  stats = {}
  stats['len'] = len(trace)
  stats['unique addr'] = len(addr_unique)
  stats['unique deltas'] = len(delta_unique)
  stats['rare deltas (< 10)'] = len(rare_deltas)
  stats['unique deltas (no rare)'] = len(delta_unique) - len(rare_deltas)
  stats['rare deltas fract'] = (len(trace) - rare_deltas.sum()) / len(trace)
  stats['deltas 50% mass'] = ceil(len(delta_unique) / 2)
  stats['deltas 50K coverage'] = 50000 / len(delta_unique)
  return stats

@click.command()
@click.option('--dataset-dir', required=True)
@click.option('--grid-size', default=50, type=int)
def preproc_stats(dataset_dir, grid_size):
  # Load data
  data = pd.read_feather(f'{dataset_dir}/trace_with_miss.feather')
  deltas = data['addr'][1:].to_numpy() - data['addr'][:-1].to_numpy()
  data_with_dt = data[1:]
  data_with_dt['delta'] = deltas
  data_with_dt = data_with_dt.reset_index()

  # Filter misses
  misses = data_with_dt[data_with_dt.miss]

  # Plot
  fig, ax = plt.subplots(ncols=2, nrows=2, sharex='col', sharey='row', figsize=(7, 8))
  hb = ax[0, 0].hexbin(data_with_dt.index, data_with_dt['addr'], gridsize=grid_size, cmap='inferno', bins='log')
  fig.colorbar(hb, ax=ax[0, 0])
  hb = ax[0, 1].hexbin(misses.index, misses['addr'], gridsize=grid_size, cmap='inferno', bins='log')
  fig.colorbar(hb, ax=ax[0, 1])
  hb = ax[1, 0].hexbin(data_with_dt.index, data_with_dt['delta'], gridsize=grid_size, cmap='inferno', bins='log')
  fig.colorbar(hb, ax=ax[1, 0])
  hb = ax[1, 1].hexbin(misses.index, misses['delta'], gridsize=grid_size, cmap='inferno', bins='log')
  fig.colorbar(hb, ax=ax[1, 1])
  ax[1,0].set_xlabel('Access Number')
  ax[1,1].set_xlabel('Access Number')
  ax[0,0].set_ylabel('Address')
  ax[0,1].set_ylabel('Missed Address')
  ax[1,0].set_ylabel('Delta')
  ax[1,1].set_ylabel('Missed Delta')
  pickle.dump(fig, open(f'{dataset_dir}/hexbins.pickle', 'wb'))

  # Stats DF
  raw_stats = stats_dict(data_with_dt['addr'].to_numpy(), data_with_dt['delta'].to_numpy())
  misses_stats = stats_dict(misses['addr'].to_numpy(), misses['delta'].to_numpy())
  raw_stats_df = pd.DataFrame(raw_stats, index=['raw']).transpose()
  misses_stats_df = pd.DataFrame(misses_stats, index=['miss']).transpose()
  stats_df = pd.concat([raw_stats_df, misses_stats_df], axis=1)
  stats_df.to_csv(f'{dataset_dir}/stats.csv')

if __name__ == '__main__':
  preproc_stats() # pylint: disable=no-value-for-parameter
