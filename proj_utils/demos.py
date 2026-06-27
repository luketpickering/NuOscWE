import pandas as pa
import numpy as np
import matplotlib.pyplot as plt

from .stats import *
from .visualize import *
from .NuFast import *

simulated_events_nu_mode = pa.read_csv("simulation/neutrino_mode_events.csv")

def draw_comp_h1():

  osc_params = {
    "experimental_baseline_km": 1300,
    "s12sq": 0.31,
    "s13sq": 0.02,
    "s23sq": 0.55,
    "delta": 0.7 * np.pi,
    "Dmsq21": 7.5e-5,
    "Dmsq31": 2.5e-3
  }
  
  events_nu_nue_cc = simulated_events_nu_mode[simulated_events_nu_mode["pid_lepton"] == 11]

  start = 0
  stop = 8
  nbins = 15
  num = nbins + 1
  
  event_osc_weights_nue_app = \
          Probability_Matter_LBL(events_nu_nue_cc["E_neutrino"], osc_params, osc_channels=["nue_appearance"])
  h1 = hist1d(data=events_nu_nue_cc["E_neutrino"], weights=event_osc_weights_nue_app, bins=np.linspace(start=start, stop=stop, num=num))

  osc_params["Dmsq31"] = 2.8e-3

  event_osc_weights_nue_app = \
          Probability_Matter_LBL(events_nu_nue_cc["E_neutrino"], osc_params, osc_channels=["nue_appearance"])
  h2 = hist1d(data=events_nu_nue_cc["E_neutrino"], weights=event_osc_weights_nue_app, bins=np.linspace(start=start, stop=stop, num=num))
  
  drawhist1d(hist=h1, color="#AA3377", lw=2, label=r"Hist A")
  drawhist1d(hist=h2, color="#4477AA", lw=2, label=r"Hist B")

  plt.xlabel(r"$x$", size="x-large")
  plt.ylabel(r"Count (Arbitrary Units)", size="x-large")
  plt.legend(fontsize="large")
  
  plt.tight_layout()
  plt.show()

def draw_comp_h2():

  osc_params = {
    "experimental_baseline_km": 1300,
    "s12sq": 0.31,
    "s13sq": 0.02,
    "s23sq": 0.55,
    "delta": 0.7 * np.pi,
    "Dmsq21": 7.5e-5,
    "Dmsq31": 2.5e-3
  }
  
  events_nu_nue_cc = simulated_events_nu_mode[simulated_events_nu_mode["pid_lepton"] == 11]

  start = 0
  stop = 8
  nbins = 15
  num = nbins + 1
  
  event_osc_weights_nue_app = \
          Probability_Matter_LBL(events_nu_nue_cc["E_neutrino"], osc_params, osc_channels=["nue_appearance"])
  h1 = hist1d(data=events_nu_nue_cc["E_neutrino"], weights=event_osc_weights_nue_app, bins=np.linspace(start=start, stop=stop, num=num))

  osc_params["s13sq"] = 0.021

  event_osc_weights_nue_app = \
          Probability_Matter_LBL(events_nu_nue_cc["E_neutrino"], osc_params, osc_channels=["nue_appearance"])
  h3 = hist1d(data=events_nu_nue_cc["E_neutrino"], weights=event_osc_weights_nue_app, bins=np.linspace(start=start, stop=stop, num=num))
  
  drawhist1d(hist=h1, color="#AA3377", lw=2, label=r"Hist A")
  drawhist1d(hist=h3, color="#EE6677", lw=2, label=r"Hist C")

  plt.xlabel(r"$x$", size="x-large")
  plt.ylabel(r"Count (Arbitrary Units)", size="x-large")
  plt.legend(fontsize="large")
  
  plt.tight_layout()
  plt.show()

def draw_comp_h3():

  osc_params = {
    "experimental_baseline_km": 1300,
    "s12sq": 0.31,
    "s13sq": 0.02,
    "s23sq": 0.55,
    "delta": 0.7 * np.pi,
    "Dmsq21": 7.5e-5,
    "Dmsq31": 2.5e-3
  }
  
  events_nu_nue_cc = simulated_events_nu_mode[simulated_events_nu_mode["pid_lepton"] == 11]

  start = 0
  stop = 8
  nbins = 15
  num = nbins + 1

  bins = np.linspace(start=start, stop=stop, num=num)
  
  event_osc_weights_nue_app = \
          Probability_Matter_LBL(events_nu_nue_cc["E_neutrino"], osc_params, osc_channels=["nue_appearance"])
  h1 = hist1d(data=events_nu_nue_cc["E_neutrino"], weights=event_osc_weights_nue_app, bins=bins)

  osc_params["Dmsq31"] = 2.8e-3

  event_osc_weights_nue_app = \
          Probability_Matter_LBL(events_nu_nue_cc["E_neutrino"], osc_params, osc_channels=["nue_appearance"])
  h2 = hist1d(data=events_nu_nue_cc["E_neutrino"], weights=event_osc_weights_nue_app, bins=bins)

  osc_params["Dmsq31"] = 2.5e-3
  osc_params["s13sq"] = 0.021

  event_osc_weights_nue_app = \
          Probability_Matter_LBL(events_nu_nue_cc["E_neutrino"], osc_params, osc_channels=["nue_appearance"])
  h3 = hist1d(data=events_nu_nue_cc["E_neutrino"], weights=event_osc_weights_nue_app, bins=bins)
  
  
  fig, (axl,axr) = plt.subplots(1,2, figsize=(8, 4))

  plt.sca(axl)
  
  drawhist1d(hist=h1, color="#AA3377", lw=2, label=r"Hist A")

  bin_centers = (bins[1:] + bins[:-1])/2
  
  drawhist1d(hist=h2, color="#4477AA", lw=2, label=r"Hist B")
  axl.errorbar(bin_centers, h2[0], yerr=pois_intervals(h2[0]*10),  color="#4477AA", lw=0, elinewidth=1)

  axl.set_xlabel(r"$x$", size="x-large")
  axl.set_ylabel(r"Count (Arbitrary Units)", size="x-large")
  axl.legend(fontsize="large")

  plt.sca(axr)

  drawhist1d(hist=h1, color="#AA3377", lw=2, label=r"Hist A")
  drawhist1d(hist=h3, color="#EE6677", lw=2, label=r"Hist C")
  axr.errorbar(bin_centers, h3[0], yerr=pois_intervals(h3[0]/4),  color="#EE6677", lw=0, elinewidth=1)

  axr.set_xlabel(r"$x$", size="x-large")
  axr.set_ylabel(r"Count (Arbitrary Units)", size="x-large")
  axr.legend(fontsize="large")
  
  plt.tight_layout()
  plt.show()


def _comp_h4_binvalues():

  osc_params = {
    "experimental_baseline_km": 1300,
    "s12sq": 0.31,
    "s13sq": 0.02,
    "s23sq": 0.55,
    "delta": 0.7 * np.pi,
    "Dmsq21": 7.5e-5,
    "Dmsq31": 2.5e-3
  }
  
  events_nu_numu_cc = simulated_events_nu_mode[simulated_events_nu_mode["pid_lepton"] == 13]

  nsamples = 10000
  nevs = events_nu_numu_cc.shape[0]
  sf = nsamples/float(nevs)
  
  start = 0.25
  stop = 8
  nbins = 50
  num = nbins + 1

  bins = np.linspace(start=start, stop=stop, num=num)
  
  event_osc_weights_numu_surv = \
          Probability_Matter_LBL(events_nu_numu_cc["E_neutrino"], osc_params, osc_channels=["numu_survival"])
  h1 = hist1d(data=events_nu_numu_cc["E_neutrino"][:nsamples], weights=event_osc_weights_numu_surv[:nsamples], bins=bins)

  osc_params["Dmsq31"] = 2.65e-3

  event_osc_weights_numu_surv = \
          Probability_Matter_LBL(events_nu_numu_cc["E_neutrino"], osc_params, osc_channels=["numu_survival"])
  h2 = hist1d(data=events_nu_numu_cc["E_neutrino"], weights=event_osc_weights_numu_surv*sf, bins=bins)
  
  return h1, h2

def comp_h4_binvalues():
  h1, h2 = _comp_h4_binvalues()
  return h1[0], h2[0]

def draw_comp_h4():
  h1, h2 = _comp_h4_binvalues()
  
  drawhist1d(hist=h1, color="#AA3377", lw=2, label=r"Observation")

  bin_centers = (h1[1][1:] + h1[1][:-1])/2
  
  drawhist1d(hist=h2, color="#4477AA", lw=2, label=r"Model Prediction")
  plt.errorbar(bin_centers, h2[0], yerr=np.sqrt(h2[0]),  color="#4477AA", lw=0, elinewidth=1)

  plt.xlabel(r"$E_\nu$ [GeV]", size="x-large")
  plt.ylabel(r"Count (Arbitrary Units)", size="x-large")
  plt.legend(fontsize="large")

  plt.tight_layout()
  plt.show()