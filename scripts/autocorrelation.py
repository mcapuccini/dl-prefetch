# Imports
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def main(trace_path, out_path, n_jobs):
    # Read to pandas
    trace = pd.read_feather(trace_path)

    # Compute deltas
    trace_np = trace['addr_int'].to_numpy()
    deltas_np = trace_np[1:] - trace_np[:-1]

    # Compute autocorrelation (needs a lot of mem)
    def autocorrelation(lag):
        coef = np.corrcoef(deltas_np[:-lag], deltas_np[lag:])[0, 1]
        return [lag, coef]
    to_plot = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(autocorrelation)(lag+1) for lag in range(500))

    # Save to disk
    np.save(out_path, to_plot)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--n-jobs", default=10, type=int)
    args = parser.parse_args()

    # Run
    main(args.trace_path, args.out_path, args.n_jobs)
