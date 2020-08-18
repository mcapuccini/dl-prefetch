# Imports
import click
import numpy as np
import pandas as pd

@click.command()
@click.option('--dataset-dir', required=True)
@click.option('--limit/--no-limit', default=False)
def preproc_feather(dataset_dir, limit):
  # Load data
  if limit:
    count=int(1e9)
  else:
    count=-1
  trace = np.fromfile(f'{dataset_dir}/roitrace.bin', dtype=np.int64, count=count)
  pc = np.fromfile(f'{dataset_dir}/pc.bin', dtype=np.int64, count=count)
  # Store as feather
  to_store = pd.DataFrame(trace, columns=['addr'])
  to_store['pc'] = pc
  to_store.to_feather(f'{dataset_dir}/trace.feather')

if __name__ == '__main__':
  preproc_feather() # pylint: disable=no-value-for-parameter