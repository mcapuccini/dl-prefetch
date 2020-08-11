import numpy as np
import pandas as pd
import click

@click.command()
@click.option('--roitrace-path', default='data/canneal_test/roitrace.bin')
@click.option('--pc-path', default='data/canneal_test/pc.bin')
@click.option('--text-path', default='data/canneal_test/roitrace.txt')
def roitrace_test(roitrace_path, pc_path, text_path):
  # Load bin files
  roitrace_bin = np.fromfile(roitrace_path, dtype=np.int64)
  pc_bin = np.fromfile(pc_path, dtype=np.int64)
  # Load text file
  roitrace_txt_hex = pd.read_csv(text_path, header=None)[0].to_numpy()
  pc_txt_hex = pd.read_csv(text_path, header=None)[1].to_numpy()
  # Parse hex
  roitrace_txt = np.array([int(h, 16) for h in roitrace_txt_hex])
  pc_txt = np.array([int(h, 16) for h in pc_txt_hex])
  # Test
  assert((roitrace_bin == roitrace_txt).all())
  assert((pc_bin == pc_txt).all())

if __name__ == '__main__':
  roitrace_test() # pylint: disable=no-value-for-parameter