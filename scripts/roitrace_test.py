import numpy as np
import pandas as pd
import click

@click.command()
@click.option('--bin-path', default='data/canneal_test/roitrace.bin')
@click.option('--text-path', default='data/canneal_test/roitrace.txt')
def roitrace_test(bin_path, text_path):
  trace_bin = np.fromfile(bin_path, dtype=np.int64)
  trace_txt_hex = pd.read_csv(text_path, header=None)[0].to_numpy()
  trace_txt = np.array([int(h, 16) for h in trace_txt_hex])
  assert((trace_bin == trace_txt).all())

if __name__ == '__main__':
  roitrace_test() # pylint: disable=no-value-for-parameter