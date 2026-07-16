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
  bvs = np.append(hist[0], 0)
  np.savetxt(filename, np.array([bvs, hist[1]]), delimiter=",")

def read_hist1d(filename):
  npd = np.loadtxt(filename, delimiter=",")
  return npd[0,:-1], npd[1]

def poisson_fluctuate(counts):
  rng = np.random.default_rng()
  return rng.poisson(counts)

def Poisson_N2LLH(data_hist, predicted_hist):
  pred_no0 = predicted_hist[0][data_hist[0] != 0]
  data_no0 = data_hist[0][data_hist[0] != 0]
  return 2*np.sum(pred_no0 - data_no0 + data_no0 * np.log(data_no0/pred_no0))

def Pearson_N2LLH(data_hist, predicted_hist):
  if data_hist.shape != predicted_hist.shape:
    raise RuntimeError(f"Shape mismatch in Pearson_N2LLH: data: {data_hist.shape}, pred: {predicted_hist.shape}")
  datanz = data_hist[data_hist > 0]
  prednz = predicted_hist[data_hist > 0]
  return np.sum(np.power(prednz - datanz, 2)/datanz)
