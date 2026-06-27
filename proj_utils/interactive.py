import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, Label, VBox, HBox, Layout
import pandas as pa

from .NuFast import *
from .stats import *

InteractiveOscProbPlot_fig = None
def InteractiveOscProbPlot(osc_params):
  global InteractiveOscProbPlot_fig
  if InteractiveOscProbPlot_fig is not None:
    plt.close(InteractiveOscProbPlot_fig)

  colors = {
    "numu": "#AA3377",
    "nue": "#4477AA",
    "antinumu": "#EE6677",
    "antinue": "#66CCEE",
  }
  
  Energy_Distribution_GeV = osc_params["experimental_energy_distribution"]
  
  Min_E_GeV = Energy_Distribution_GeV["peak"] - 3 * Energy_Distribution_GeV["width"]
  Min_E_GeV = Min_E_GeV if Min_E_GeV > 0 else Energy_Distribution_GeV["peak"]/5
  Max_E_GeV = Energy_Distribution_GeV["peak"] + 3 * Energy_Distribution_GeV["width"]

  Es = np.linspace(Min_E_GeV, Max_E_GeV,1000)
  
  var_params = osc_params.copy()
  
  numu_surv_prob, antinumu_surv_prob, nue_app_prob, antinue_app_prob = \
    Probability_Matter_LBL(Es, var_params, 
                           osc_channels=["numu_survival", "antinumu_survival", "nue_appearance", "antinue_appearance"])

  plt.ioff()
  InteractiveOscProbPlot_fig, axes = plt.subplots(nrows=2, ncols=2, \
                                                  gridspec_kw=dict(left=0.1,bottom=0.1, top=0.99, wspace=0.3,hspace=0.3), \
                                                  figsize=(8,5))
  InteractiveOscProbPlot_fig.canvas.toolbar_visible = False
  
  axpl, axpr, axel, axer = axes.flatten()
  
  alpha = 0.2

  nu_unosc_evr = gaus(Es, loc=Energy_Distribution_GeV["peak"], scale=Energy_Distribution_GeV["width"]) \
                   /gaus(Energy_Distribution_GeV["peak"], loc=Energy_Distribution_GeV["peak"], scale=Energy_Distribution_GeV["width"])
  antinu_unosc_evr = nu_unosc_evr*0.3

  numu_unosc_evr_lo_l = axpl.fill_between(Es, nu_unosc_evr*0.9, np.zeros_like(Es), fc="grey", alpha=0.1)
  numu_surv_prob_lo, = axpl.plot(Es, numu_surv_prob, c=colors["numu"], alpha=alpha)
  numu_surv_prob_l, = axpl.plot(Es, numu_surv_prob, c=colors["numu"], label=r"muon neutrino")
  antinumu_surv_prob_lo, = axpl.plot(Es, antinumu_surv_prob, c=colors["antinumu"], alpha=alpha, ls="dashed")
  antinumu_surv_prob_l, = axpl.plot(Es, antinumu_surv_prob, c=colors["antinumu"], ls="dashed", label=r"muon antineutrino")
  axpl.set_xlabel(r"$E_{\nu}$ [GeV]", size="large")
  axpl.set_ylabel(r"$P_{\nu_\mu \rightarrow \nu_\mu}$", size="x-large")
  axpl.set_ylim([0,1.1])
  axpl.legend()
  
  numu_unosc_evr_lo_r = axpr.fill_between(Es, nu_unosc_evr*0.3*0.9, np.zeros_like(Es), fc="grey", alpha=0.1)
  nue_surv_prob_lo, = axpr.plot(Es, nue_app_prob, c=colors["nue"], alpha=alpha)
  nue_surv_prob_l, = axpr.plot(Es, nue_app_prob, c=colors["nue"], label=r"electron neutrino")
  antinue_surv_prob_lo, = axpr.plot(Es, antinue_app_prob, c=colors["antinue"], alpha=alpha, ls="dashed")
  antinue_surv_prob_l, = axpr.plot(Es, antinue_app_prob, c=colors["antinue"], ls="dashed", label=r"electron antineutrino")
  axpr.set_xlabel(r"$E_{\nu}$ [GeV]", size="large")
  axpr.set_ylabel(r"$P_{\nu_\mu \rightarrow \nu_\mathrm{e}}$", size="x-large")
  axpr.set_ylim([0,0.3])
  axpr.legend()

  numu_surv_evr_lo, = axel.plot(Es, numu_surv_prob*nu_unosc_evr, c=colors["numu"], alpha=alpha)
  numu_surv_evr_l, = axel.plot(Es, numu_surv_prob*nu_unosc_evr, c=colors["numu"], label=r"muon neutrino")
  nue_app_evr_lo, = axel.plot(Es, nue_app_prob*nu_unosc_evr, c=colors["nue"], alpha=alpha)
  nue_app_evr_l, = axel.plot(Es, nue_app_prob*nu_unosc_evr, c=colors["nue"], label=r"electron neutrino")
  axel.set_xlabel(r"$E_{\nu}$ [GeV]", size="large")
  axel.set_ylabel(r"Observed Neutrinos")
  axel_ymax = np.max(numu_surv_prob*nu_unosc_evr)*1.1
  axel.set_ylim([0, axel_ymax])
  axel.legend()

  antinumu_surv_evr_lo, = axer.plot(Es, antinumu_surv_prob*antinu_unosc_evr, c=colors["antinumu"], alpha=alpha, ls="dashed")
  antinumu_surv_evr_l, = axer.plot(Es, antinumu_surv_prob*antinu_unosc_evr, c=colors["antinumu"], ls="dashed", label=r"muon antineutrino")
  antinue_app_evr_lo, = axer.plot(Es, antinue_app_prob*antinu_unosc_evr, c=colors["antinue"], alpha=alpha, ls="dashed")
  antinue_app_evr_l, = axer.plot(Es, antinue_app_prob*antinu_unosc_evr, c=colors["antinue"], ls="dashed", label=r"electron antineutrino")
  axer.set_xlabel(r"$E_{\nu}$ [GeV]", size="large")
  axer.set_ylabel(r"Observed Antineutrinos")
  axer.set_ylim([0, axel_ymax])
  axer.legend()

  controls_spec = [
    dict( pname = "Dmsq31", srange = [2.3, 2.7], sf = 1E3, desc = r"$\Delta\mathrm{m}_{31}^{2}$ [$10^{-3}$ eV]" ),
    dict( pname = "Dmsq21", srange = [6, 9], sf = 1E5, desc = r"$\Delta\mathrm{m}_{21}^{2}$ [$10^{-5}$ eV]" ),
    dict( pname = "delta", srange = [-1, 1], sf = 1.0/np.pi, desc = r"$\delta_\mathrm{CP}/\pi$" ),

    dict( pname = "s12sq", srange = [2.5, 3.6], sf = 10, desc = r"$\mathrm{sin}^{2}(\theta_{12})$ [$10^{-1}$]" ),
    dict( pname = "s13sq", srange = [1, 2.5], sf = 1E2, desc = r"$\mathrm{sin}^{2}(\theta_{13})$ [$10^{-2}$]" ),
    dict( pname = "s23sq", srange = [4, 7], sf = 10, desc = r"$\mathrm{sin}^{2}(\theta_{23})$ [$10^{-1}$]" ),
  ]

  sliders = []
  controls = []

  for c in controls_spec:
    sliders.append(FloatSlider(orientation='horizontal',
                               value=osc_params[c["pname"]]*c["sf"],
                               min=c["srange"][0],
                               max=c["srange"][1],
                               step=(c["srange"][1]-c["srange"][0])/100
                              ))
    controls.append(HBox([Label(c["desc"]), sliders[-1]]))
  
  def update(val):
    for i,c in enumerate(controls_spec):
      var_params[c["pname"]] = sliders[i].value / c["sf"]
      
    numu_surv_prob, antinumu_surv_prob, nue_app_prob, antinue_app_prob = \
        Probability_Matter_LBL(Es, var_params, 
                               osc_channels=["numu_survival", "antinumu_survival", "nue_appearance", "antinue_appearance"])
    
    numu_surv_prob_l.set_ydata(numu_surv_prob)
    antinumu_surv_prob_l.set_ydata(antinumu_surv_prob)
    nue_surv_prob_l.set_ydata(nue_app_prob)
    antinue_surv_prob_l.set_ydata(antinue_app_prob)
    
    numu_surv_evr_l.set_ydata(numu_surv_prob*nu_unosc_evr)
    nue_app_evr_l.set_ydata(nue_app_prob*nu_unosc_evr)
    antinumu_surv_evr_l.set_ydata(antinumu_surv_prob*antinu_unosc_evr)
    antinue_app_evr_l.set_ydata(antinue_app_prob*antinu_unosc_evr)
    
    InteractiveOscProbPlot_fig.canvas.draw_idle()

  for s in sliders:
    s.observe(update)

  control_container = HBox([
    VBox([controls[i] for i in range(3)],layout=Layout(align_items="flex-end")),
    VBox([controls[i+3] for i in range(3)],layout=Layout(align_items="flex-end")),
  ],layout=Layout(justify_content="flex-start"))
  
  display(InteractiveOscProbPlot_fig.canvas)
  display(control_container)

InteractiveLikelihoodScan_fig = None
def InteractiveLikelihoodScan():
  global InteractiveLikelihoodScan_fig
  if InteractiveLikelihoodScan_fig is not None:
    plt.close(InteractiveLikelihoodScan_fig)

  osc_params = {
    "experimental_baseline_km": 1300,
    "s12sq": 0.31,
    "s13sq": 0.02,
    "s23sq": 0.55,
    "delta": 0.7 * np.pi,
    "Dmsq21": 7.5e-5,
    "Dmsq31": 2.5e-3
  }
  
  colors = {
    "numu": "#AA3377",
    "nue": "#4477AA",
    "antinumu": "#EE6677",
    "antinue": "#66CCEE",
  }

  simulated_events_nu_mode = pa.read_csv("simulation/neutrino_mode_events.csv")
  events_nu_numu_cc = simulated_events_nu_mode[simulated_events_nu_mode["pid_lepton"] == 13]

  nsamples = 10000
  nevs = events_nu_numu_cc.shape[0]
  sf = nsamples/float(nevs)
  
  start = 0.25
  stop = 8
  nbins = 50
  num = nbins + 1

  bins = np.linspace(start=start, stop=stop, num=num)
  bin_centers = (bins[1:] + bins[:-1])/2 
  
  event_osc_weights_numu_surv = \
          Probability_Matter_LBL(events_nu_numu_cc["E_neutrino"], osc_params, osc_channels=["numu_survival"])
  h1 = hist1d(data=events_nu_numu_cc["E_neutrino"][:nsamples], weights=event_osc_weights_numu_surv[:nsamples], bins=bins)

  plt.ioff()
  InteractiveLikelihoodScan_fig, axes = plt.subplots(nrows=1, ncols=2, \
                                                  gridspec_kw=dict(left=0.1,bottom=0.15, top=0.99, wspace=0.3,hspace=0.3), \
                                                  figsize=(12,4))
  InteractiveLikelihoodScan_fig.canvas.toolbar_visible = False
  
  axl, axr = axes.flatten()

  axl.stairs(h1[0], h1[1], color="#AA3377", lw=2, label=r"Observation")
  
  osc_params["Dmsq31"] = 2.65e-3

  event_osc_weights_numu_surv = \
          Probability_Matter_LBL(events_nu_numu_cc["E_neutrino"], osc_params, osc_channels=["numu_survival"])
  h2 = hist1d(data=events_nu_numu_cc["E_neutrino"], weights=event_osc_weights_numu_surv*sf, bins=bins)

  pred_hist_l = axl.stairs(h2[0], h2[1], color="#4477AA", lw=2, label=r"Model Prediction")
  pred_errs_l = axl.errorbar(bin_centers, h2[0], yerr=np.sqrt(h2[0]),  color="#4477AA", lw=0, elinewidth=1)
  
  axl.set_xlabel(r"$E_{\nu}$ [GeV]", size="large")
  axl.set_ylabel(r"$P_{\nu_\mu \rightarrow \nu_\mu}$", size="x-large")
  axl.set_ylim([0,320])
  axl.legend()

  controls_spec = [
    dict( pname = "Dmsq31", srange = [2.3, 2.7], sf = 1E3, desc = r"$\Delta\mathrm{m}_{31}^{2}$ [$10^{-3}$ eV]" ),
  ]

  lhood_x = np.linspace(controls_spec[0]["srange"][0],controls_spec[0]["srange"][1], 51)
  lhood_y = np.zeros_like(lhood_x)

  lhood_l, = axr.plot(lhood_x,lhood_y, ".")
  
  axr.set_xlabel(controls_spec[0]["desc"], size="large")
  axr.set_ylabel(r"$-2\log(\mathcal{L})$", size="x-large")
  axr.set_ylim([-5,1000])
  
  sliders = []
  controls = []

  for c in controls_spec:
    sliders.append(FloatSlider(orientation='horizontal',
                               value=osc_params[c["pname"]]*c["sf"],
                               min=c["srange"][0],
                               max=c["srange"][1],
                               step=(c["srange"][1]-c["srange"][0])/50
                              ))
    controls.append(HBox([Label(c["desc"]), sliders[-1]]))

  var_params = osc_params.copy()
  def update(val):
    for i,c in enumerate(controls_spec):
      var_params[c["pname"]] = sliders[i].value / c["sf"]
      
    event_osc_weights_numu_surv = \
          Probability_Matter_LBL(events_nu_numu_cc["E_neutrino"], var_params, osc_channels=["numu_survival"])
    h2 = hist1d(data=events_nu_numu_cc["E_neutrino"], weights=event_osc_weights_numu_surv*sf, bins=bins)

    nonlocal pred_hist_l, pred_errs_l
    pred_hist_l.remove()
    pred_errs_l.remove()
    
    pred_hist_l = axl.stairs(h2[0], h2[1], color="#4477AA", lw=2, label=r"Model Prediction")
    pred_errs_l = axl.errorbar(bin_centers, h2[0], yerr=np.sqrt(h2[0]),  color="#4477AA", lw=0, elinewidth=1)

    lhood_i = np.argmax(lhood_x > sliders[i].value)
    lhood_y[lhood_i] = Pearson_N2LLH(h1[0],h2[0])

    lhood_l.set_ydata(lhood_y)
    
    InteractiveLikelihoodScan_fig.canvas.draw_idle()

  for s in sliders:
    s.observe(update)
  
  display(InteractiveLikelihoodScan_fig.canvas)
  display(controls[0])