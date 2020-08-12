# Imports
import click
import numpy as np
import pandas as pd
from cachesim import Cache, CacheSimulator, MainMemory
from tqdm import trange

@click.command()
@click.option('--dataset-dir', required=True)
def preproc_misses(dataset_dir):
  # Intel inclusive cache simulator
  cacheline_size = 64
  mem = MainMemory()

  l3 = Cache(name="L3",
             sets=20480,
             ways=16,
             cl_size=cacheline_size,
             replacement_policy="LRU",
             write_back=True,
             write_allocate=True,
             store_to=None,
             load_from=None,
             victims_to=None,
             swap_on_load=False)
  l2 = Cache(name="L2",
             sets=512,
             ways=8,
             cl_size=cacheline_size,
             replacement_policy="LRU",
             write_back=True,
             write_allocate=True,
             store_to=l3,
             load_from=l3,
             victims_to=None,
             swap_on_load=False)
  l1 = Cache(name="L1",
             sets=64,
             ways=8,
             cl_size=cacheline_size,
             replacement_policy="LRU",
             write_back=True,
             write_allocate=True,
             store_to=l2,
             load_from=l2,
             victims_to=None,
             swap_on_load=False) # inclusive/exclusive does not matter in first-level

  mem.load_to(l3)
  mem.store_from(l3)
  cs = CacheSimulator(first_level=l1, main_memory=mem)

  # Load data
  trace = np.fromfile(f'{dataset_dir}/roitrace.bin', dtype=np.int64)
  pc = np.fromfile(f'{dataset_dir}/pc.bin', dtype=np.int64)

  # Run simulation and track misses
  miss = np.zeros(len(trace), dtype=np.bool_)
  miss_count = 0
  for i in trange(len(trace), desc='cache simulator'):
    cs.load(trace[i])
    if (l1.MISS_count > miss_count):
      miss[i] = True
      miss_count += 1
      assert (miss_count == l1.MISS_count)
  assert (miss.sum() == l1.MISS_count)

  # Store
  to_store = pd.DataFrame(trace, columns=['addr'])
  to_store['pc'] = pc
  to_store['miss'] = miss
  to_store.to_feather(f'{dataset_dir}/trace_with_miss.feather')

if __name__ == '__main__':
  preproc_misses() # pylint: disable=no-value-for-parameter