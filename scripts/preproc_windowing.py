# Imports
import numpy as np
import torch
import click

@click.command()
@click.option('--dataset-dir', required=True)
@click.option('--npz-archive', required=True)
@click.option('--series', required=True)
@click.option('--lookback', default=64, type=int)
@click.option('--out-name', required=True)
def preproc_windowing(
  dataset_dir,
  npz_archive,
  series,
  lookback,
  out_name,
):
  # Load
  series = np.load(f'{dataset_dir}/{npz_archive}')[series]
  
  # Windowing
  with torch.no_grad():
    series_torch = torch.from_numpy(series)
    windowed_series = series_torch.unfold(0, lookback, 1)
    torch.save(windowed_series, f'{dataset_dir}/{out_name}.{lookback}.pt')

if __name__ == '__main__':
  preproc_windowing() # pylint: disable=no-value-for-parameter