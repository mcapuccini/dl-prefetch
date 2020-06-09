# Imports
import argparse
import string

import numpy as np
from joblib import Parallel, delayed
from pyspark.sql import Row, SparkSession


def main(trace_path, out_path, spark_master, spark_driver_memory, spark_driver_max_result_size, n_jobs):
    # Start Spark
    spark = SparkSession \
        .builder \
        .master(spark_master) \
        .config("spark.driver.memory", spark_driver_memory) \
        .config("spark.driver.maxResultSize", spark_driver_max_result_size) \
        .config("spark.sql.execution.arrow.enabled", "true") \
        .getOrCreate()

    # Load traces
    trace_df = spark.read.csv(
        trace_path,
        sep=' ',
        header=None,
    )

    # Filter malformed data
    addressesRDD = trace_df \
        .select("_c2") \
        .rdd \
        .filter(lambda x: isinstance(x._c2, str)) \
        .filter(lambda x: all(c in string.hexdigits for c in x._c2[-2:]))

    # Parse addresses
    addresses_df = addressesRDD.map(lambda x: Row(addr=int(x._c2, 16))).toDF()

    # Convert to pandas
    addresses = addresses_df.toPandas()

    # Stop Spark
    spark.stop()

    # Compute deltas
    address_np = addresses['addr'].to_numpy()
    deltas_np = address_np[1:] - address_np[:-1]

    # Compute autocorrelation (needs a lot of mem)
    def autocorrelation(lag):
        coef = np.corrcoef(deltas_np[:-lag], deltas_np[lag:])[0, 1]
        return [lag, coef]
    to_plot = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(autocorrelation)(lag+1) for lag in range(500))

    # Save to disk
    np.save(to_plot, out_path)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--spark-master", default="local[*]", required=True)
    parser.add_argument("--spark-driver-memory", default="100G", required=True)
    parser.add_argument("--spark-driver-max-result-size",
                        default="50G", required=True)
    parser.add_argument("--n-jobs", default=10, required=True)
    args = parser.parse_args()

    # Run
    main(args.trace_path, args.out_path, args.spark_master,
         args.spark_driver_memory, args.spark_driver_max_result_size, args.n_jobs)
