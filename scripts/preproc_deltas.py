# Imports
import click
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

@click.command()
@click.option('--dataset-dir', required=True)
def preproc_deltas(dataset_dir):
  # Load data
  trace = pd.read_feather(f'{dataset_dir}/trace_with_miss.feather')
  addr = trace['addr'].to_numpy()
  misses = trace['miss'].to_numpy()

  # Compute deltas
  deltas = np.append([0], addr[1:] - addr[:-1])
  miss_deltas = np.append([0], addr[misses][1:] - addr[misses][:-1])

  # Store
  trace['delta'] = deltas
  trace['delta_miss'] = 0
  trace['delta_miss'][misses] = miss_deltas
  trace.to_feather(f'{dataset_dir}/trace_with_deltas.feather')

if __name__ == '__main__':
  preproc_deltas() # pylint: disable=no-value-for-parameter