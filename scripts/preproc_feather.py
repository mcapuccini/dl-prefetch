# Imports
import click
import numpy as np
import pandas as pd

@click.command()
@click.option('--dataset-dir', required=True)
def preproc_feather(dataset_dir):
  # Load data and compute deltas
  trace = np.fromfile(f'{dataset_dir}/roitrace.bin', dtype=np.int64)
  pc = np.fromfile(f'{dataset_dir}/pc.bin', dtype=np.int64)
  deltas = trace[1:] - trace[:-1]
  # Store as feather
  to_store = pd.DataFrame(trace[1:], columns=['addr'])
  to_store['pc'] = pc[1:]
  to_store['deltas'] = deltas
  to_store.to_feather(f'{dataset_dir}/trace.feather')

if __name__ == '__main__':
  preproc_feather() # pylint: disable=no-value-for-parameter