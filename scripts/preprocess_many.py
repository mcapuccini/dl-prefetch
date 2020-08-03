from pathlib import Path

import click

from preprocessing import preprocessing

@click.command()
@click.option('--traces-path', required=True)
@click.option('--spark-master', default='local[*]')
@click.option('--spark-driver-memory', default='100G')
@click.option('--spark-driver-max-result-size', default='50G')
@click.option('--test-size', default=500000, type=int)
def preprocess_many(
  traces_path,
  spark_master,
  spark_driver_memory,
  spark_driver_max_result_size,
  test_size,
):
  pathlist = Path(traces_path).glob('**/roitrace.csv')
  for trace_path in pathlist:
    trace_dir = trace_path.parent
    # preprocess if feather files don't exist
    if not trace_dir.joinpath('train.feather').exists() and \
      not trace_dir.joinpath('test.feather').exists():
      print(f'Preprocessing {str(trace_dir)} ...')
      preprocessing(
        trace_path,
        trace_path,
        spark_master,
        spark_driver_memory,
        spark_driver_max_result_size,
        test_size,
      )
    else:
      print(f'Skipping {str(trace_dir)}')

if __name__ == '__main__':
  preprocess_many() # pylint: disable=no-value-for-parameter
