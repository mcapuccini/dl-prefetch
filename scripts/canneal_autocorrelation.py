# %% Imports
import pandas as pd
import matplotlib.pyplot as plt

# %% Load traces
trace = pd.read_csv(
    '/traces/canneal/pinatrace.out',
    sep=' ',
    header=None,
    skiprows=100000, 
    nrows=200000
)
trace

# %% Get addresses
addresses = trace[2][:-1].map(lambda x: int(x, base=16)).to_numpy()
addresses

# %% Compute deltas
deltas = pd.Series(addresses[1:] - addresses[:-1])
deltas

# %% Plot autocorrelation
to_plot = []
for l in range(0,501):
    to_plot.append(deltas.autocorr(lag=l))
plt.plot(to_plot)
