# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, Row
import string
from pyspark.sql.window import Window
import numpy as np
from joblib import Parallel, delayed

# %% Start Spark
spark = SparkSession \
    .builder \
    .master("local[*]") \
    .config("spark.driver.memory", "100G") \
    .config("spark.driver.maxResultSize", "50G") \
    .config("spark.sql.execution.arrow.enabled", "true") \
    .getOrCreate()

# %% Load traces
traceDF = spark.read.csv(
    '/traces/canneal/pinatrace.out',
    sep=' ',
    header=None,
)
traceDF.head()

# %% Count entries
traceDF.count()

# %% Filter malformed data
addressesRDD = traceDF\
    .select("_c2") \
    .rdd \
    .filter(lambda x: isinstance(x._c2, str) ) \
    .filter(lambda x: all(c in string.hexdigits for c in x._c2[-2:])) \
    .cache()

# %% Count entries
addressesRDD.count()

# %% Parse addresses
addressesDF = addressesRDD.map(lambda x: Row(addr=int(x._c2, 16))).toDF()
addressesDF.head()

# %% Convert to pandas
addresses = addressesDF.toPandas()
addresses

# %% Stop Spark
spark.stop()

# %% Compute deltas
addressesNP = addresses['addr'].to_numpy()
deltasNP = addressesNP[1:] - addressesNP[:-1]
deltasNP

# %% Compute autocorrelation (needs a lot of mem)
def autocorrelation(lag):
    coef = np.corrcoef(deltasNP[:-lag],deltasNP[lag:])[0,1]
    return [lag, coef]
to_plot = Parallel(n_jobs=10, backend="threading")(delayed(autocorrelation)(lag+1) for lag in range(500))
to_plot

# %% Plot autocorrelation
plt.scatter(np.array(to_plot)[:,0], np.array(to_plot)[:,1])
