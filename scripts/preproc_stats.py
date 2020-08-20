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

  # Stats
  stats = {}
  stats['trace len'] = len(trace)
  stats['unique addr'] = len(addr_unique)
  stats['unique deltas'] = len(delta_unique)

  # 50 mass and 50K coverage
  delta_50_mass = (delta_counts.cumsum() < (len(deltas) / 2)).sum() + 1
  stats['deltas 50% mass'] = delta_50_mass
  stats['deltas 50K coverage'] = delta_counts[:50000].sum() / len(trace)

  # Rare deltas < 10
  rare_deltas = delta_counts[delta_counts < 10]
  rare_deltas_sum = rare_deltas.sum()
  stats['rare deltas (< 10)'] = rare_deltas_sum
  stats['rare deltas fract'] = rare_deltas_sum / len(trace)
  stats['unique rare deltas'] = len(rare_deltas)
  stats['unique rare deltas fract'] = (len(delta_unique) - len(rare_deltas)) / len(rare_deltas)
  stats['unique deltas (no rare)'] = len(delta_unique) - len(rare_deltas)

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
  addr_misses = addr[misses]
  deltas_misses = data['delta_miss'][misses].to_numpy()
  delta_accmiss = deltas[misses]
  misses_idx = np.where(misses)[0]

  # Compute histograms
  if (histograms):
    time_dt = np.histogram2d(range(len(deltas)), deltas, bins=bins)
    missn_dtmiss = np.histogram2d(range(misses.sum()), deltas_misses, bins=bins)
    missn_dt = np.histogram2d(range(misses.sum()), delta_accmiss, bins=bins)
    time_dt_miss = np.histogram2d(misses_idx, delta_accmiss, bins=bins)
    np.savez(
      f'{dataset_dir}/histograms.npz',
      time_dt=time_dt,
      missn_dtmiss=missn_dtmiss,
      missn_dt=missn_dt,
      time_dt_miss=time_dt_miss
    )

  # Compute stats
  access_stats = stats_dict(addr, deltas)
  access_stats_df = pd.DataFrame(access_stats, index=['access']).transpose()

  misses_stats = stats_dict(addr_misses, deltas_misses)
  misses_stats_df = pd.DataFrame(misses_stats, index=['miss']).transpose()

  accmiss_stats = stats_dict(addr_misses, delta_accmiss)
  accmiss_stats_df = pd.DataFrame(accmiss_stats, index=['miss (access delta)']).transpose()

  stats_df = pd.concat([access_stats_df, misses_stats_df, accmiss_stats_df], axis=1)
  stats_df.to_csv(f'{dataset_dir}/stats.csv')

if __name__ == '__main__':
  preproc_stats() # pylint: disable=no-value-for-parameter
