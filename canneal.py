# %% Imports
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# %% Load and preprocess data
def toInt(x): # helper to conver hex string to int
    return int(x.numpy(), base=16)

canneal = tf.data.experimental.CsvDataset(['/disk/traces/canneal_test.6.0'], [tf.string], field_delim=' ', select_cols=[0])
canneal = canneal.map(lambda x: tf.py_function(toInt, [x], [tf.int64]))
canneal = canneal.batch(2, drop_remainder=True)
canneal = canneal.map(lambda x: float(x[1] - x[0]))
canneal_batch = next(iter(canneal.batch(200000).skip(1)))

# %% Compute autocorrelation
auto_correlation = tfp.stats.auto_correlation(canneal_batch, max_lags=500)

# %% Plot it
plt.plot(auto_correlation.numpy().T)
