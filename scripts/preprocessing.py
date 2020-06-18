# Imports
import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import PandasUDFType, col, pandas_udf


# Helper
@pandas_udf('long', PandasUDFType.SCALAR)
def parse_hex(series):
    return series.map(lambda x: int(x, base=16))


def main(trace_path, out_dir, spark_master, spark_driver_memory, spark_driver_max_result_size, clip_size, test_dev_size):
    # Start Spark
    spark = SparkSession \
        .builder \
        .master(spark_master) \
        .config('spark.driver.memory', spark_driver_memory) \
        .config('spark.driver.maxResultSize', spark_driver_max_result_size) \
        .config('spark.sql.execution.arrow.enabled', 'true') \
        .getOrCreate()

    # Load trace
    trace = spark.read \
        .format('csv') \
        .option('header', 'true') \
        .load(trace_path) \
        .withColumn('pc_int', parse_hex(col('pc'))) \
        .withColumn('addr_int', parse_hex(col('addr'))) \
        .drop(col('pc')) \
        .drop(col('addr')) \
        .toPandas()

    # Train, test, dev split
    trace_clip = trace[clip_size:][:-clip_size]\
        .reset_index(drop=True)
    trace_train = trace_clip[:-(test_dev_size*2)]\
        .reset_index(drop=True)
    trace_test = trace_clip[-test_dev_size:]\
        .reset_index(drop=True)
    trace_dev = trace_clip[:-test_dev_size][-test_dev_size:]\
        .reset_index(drop=True)

    # Save as feather file
    trace_train.to_feather(out_dir + '/traces/canneal/train.feather')
    trace_test.to_feather(out_dir + '/traces/canneal/test.feather')
    trace_dev.to_feather(out_dir + '/traces/canneal/dev.feather')


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-path', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--spark-master', default='local[*]')
    parser.add_argument('--spark-driver-memory', default='100G')
    parser.add_argument('--spark-driver-max-result-size', default='50G')
    parser.add_argument('--clip-size', default=100000, type=int)
    parser.add_argument('--test-dev-size', default=500000, type=int)
    args = parser.parse_args()

    # Run
    main(args.trace_path, args.out_dir, args.spark_master, args.spark_driver_memory,
         args.spark_driver_max_result_size, args.clip_size, args.test_dev_size)
