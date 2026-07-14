import numpy as np
import matplotlib.pyplot as plt

from .stats import gaus, hist1d

def get_scaled_gaus(num_samples, loc, scale, bins):
  bin_width = (bins[-1] - bins[0])/(bins.shape[0]-1)
  g_scaling_factor = bin_width * num_samples
  values = np.linspace(bins[0], bins[-1], 1000)
  return (values, gaus(values, loc=loc, scale=scale)*g_scaling_factor)

def drawgaus(num_samples, loc, scale, bins, **kwargs):
  x,g = get_scaled_gaus(num_samples, loc, scale, bins)
  label = r"$g(x)$" if "label" not in kwargs else kwargs["label"]
  return plt.plot(x, g, label=label)

def drawhist1d(*, hist=None, data=None, bins=None, weights=None, **kwargs):
    if data is not None:
        counts, bins = hist1d(data, bins, weights)
    elif hist is not None:
        counts = hist[0]
        bins = hist[1]
    else:
        raise RuntimeError("Must pass either hist= or data= arguments")

    return plt.gca().stairs(counts, bins, **kwargs)

def hist1dtoline(bin_vals, bins):
  vals = np.zeros((2, (bin_vals.shape[0] + 1) * 2))
  vals[0,0] = bins[0]
  vals[1,0] = 0
  for i in range(bin_vals.shape[0]):
    vals[0,1 + 2*i] = bins[i]
    vals[1,1 + 2*i] = bin_vals[i]
    
    vals[0,2 + 2*i] = bins[i+1]
    vals[1,2 + 2*i] = bin_vals[i]

  vals[0,-1] = bins[-1]
  vals[1,-1] = 0

  return vals