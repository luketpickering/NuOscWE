import numpy as np
from scipy.optimize import newton

def _pois_interval_Wald(c):
    return c - newton(lambda ll: ll + np.sqrt(ll) - c,c), newton(lambda lh: lh - np.sqrt(lh) - c,c) - c

def pois_intervals(counts):
    l = []
    h = []
    for c in counts:
      a, b = _pois_interval_Wald(c)
      l.append(a)
      h.append(b)
    return [l,h]

def gaus(x, loc, scale):
    return (1.0/(scale*np.sqrt(2*np.pi))) * np.exp(-np.power(x-loc,2)/(2*np.power(scale,2)))  

def hist1d(data, bins, weights=None):
    if weights is None:
        bin_vals, bins = np.histogram(data, bins=bins) 
    else:
        bin_vals, bins = np.histogram(data, bins=bins, weights=weights)
    return (bin_vals, bins)

def write_hist1d(hist, filename):
  np.save(filename, np.array([bin_vals, bins]))

def read_hist1d(filename):
  npd = numpy.load(filename)
  return npd[0], ndp[1]

def poisson_fluctuate(counts):
    rng = np.random.default_rng()
    return rng.poisson(counts)

def Poisson_N2LLH(data_hist, predicted_hist):
    pred_no0 = predicted_hist[0][data_hist[0] != 0]
    data_no0 = data_hist[0][data_hist[0] != 0]
    return 2*np.sum(pred_no0 - data_no0 + data_no0 * np.log(data_no0/pred_no0))

def Pearson_N2LLH(data_hist, predicted_hist):
    return np.sum(np.power(predicted_hist - data_hist,2)/data_hist)
