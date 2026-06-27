import numpy as np
import matplotlib.pyplot as plt

from .stats import gaus

def drawgaus(num_samples, loc, scale, bins):
  bin_width = (bins[-1] - bins[0])/(bins.shape[0]-1)
  g_scaling_factor = bin_width * num_samples
  values = np.linspace(bins[0], bins[-1], 1000)
  plt.plot(values, gaus(values, loc=loc, scale=scale)*g_scaling_factor, label=r"$g(x)$")

def drawhist1d(*, hist=None, data=None, bins=None, weights=None, **kwargs):
    if data is not None:
        counts, bins = hist1d(data, bins, weights)
    elif hist is not None:
        counts = hist[0]
        bins = hist[1]
    else:
        raise RuntimeError("Must pass either hist= or data= arguments")

    return plt.gca().stairs(counts, bins, **kwargs)