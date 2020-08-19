# Imports
import click
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

def stats_dict(trace, deltas):
  # Compute unique addr/deltas
  addr_unique = np.unique(trace)
  delta_unique, delta_counts = np.unique(deltas, return_counts=True)
  delta_counts[::-1].sort() # sort from most frequent to least frequent
  rare_deltas = delta_counts[delta_counts < 10]
  # Stats
  stats = {}
  stats['len'] = len(trace)
  stats['unique addr'] = len(addr_unique)
  stats['unique deltas'] = len(delta_unique)
  stats['unique rare deltas (< 10)'] = len(rare_deltas)
  stats['unique deltas (no rare)'] = len(delta_unique) - len(rare_deltas)
  stats['rare deltas fract'] = rare_deltas.sum() / len(trace)
  stats['deltas 50% mass'] = (delta_counts.cumsum() < (len(deltas) / 2)).sum()
  stats['deltas 50K coverage'] = delta_counts[:50000].sum() / len(trace)
  return stats

@click.command()
@click.option('--dataset-dir', required=True)
@click.option('--bins', default=100, type=int)
@click.option('--histograms/--no-histograms', default=True)
def preproc_stats(dataset_dir, bins, histograms):
  # Load data
  data = pd.read_feather(f'{dataset_dir}/trace_with_deltas.feather')
  addr = data['addr'].to_numpy()
  deltas = data['delta'].to_numpy()
  misses = data['miss'].to_numpy()
  misses_idx = np.where(misses)[0]
  addr_misses = addr[misses]
  deltas_misses = data['delta_miss'][misses].to_numpy()

  # Compute histograms
  if(histograms):
    time_addr = np.histogram2d(range(len(addr)), addr, bins=bins)
    time_dt = np.histogram2d(range(len(deltas)), deltas, bins=bins)
    time_addr_miss = np.histogram2d(misses_idx, addr_misses, bins=bins)
    time_dt_miss = np.histogram2d(misses_idx, deltas_misses, bins=bins)
    np.savez(
      f'{dataset_dir}/histograms.npz',
      time_addr=time_addr,
      time_dt=time_dt,
      time_addr_miss=time_addr_miss,
      time_dt_miss=time_dt_miss,
    )

  # Compute stats
  raw_stats = stats_dict(addr, deltas)
  misses_stats = stats_dict(addr_misses, deltas_misses)
  raw_stats_df = pd.DataFrame(raw_stats, index=['raw']).transpose()
  misses_stats_df = pd.DataFrame(misses_stats, index=['miss']).transpose()
  stats_df = pd.concat([raw_stats_df, misses_stats_df], axis=1)
  stats_df.to_csv(f'{dataset_dir}/stats.csv')

if __name__ == '__main__':
  preproc_stats() # pylint: disable=no-value-for-parameter
