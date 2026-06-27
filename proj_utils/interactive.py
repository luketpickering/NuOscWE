import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

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
  Max_E_GeV = Energy_Distribution_GeV["peak"] + 5 * Energy_Distribution_GeV["width"]

  Es = np.linspace(Min_E_GeV, Max_E_GeV,1000)
  
  var_params = osc_params.copy()
  
  numu_surv_prob, antinumu_surv_prob, nue_app_prob, antinue_app_prob = \
    Probability_Matter_LBL(Es, var_params, 
                           osc_channels=["numu_survival", "antinumu_survival", "nue_appearance", "antinue_appearance"])

  plt.ioff()
  InteractiveOscProbPlot_fig, axes = plt.subplots(nrows=2, ncols=2)
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
  axpr.set_ylabel(r"$P_{\nu_\mu \rightarrow \nu_\mathrm{e}}$", size="large")
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

  dmsq31_slider = widgets.FloatSlider(
    orientation='horizontal',
    description=r"$\Delta\mathrm{m}_{31}^{2}$ [$10^{-3}$ eV]",
    value=osc_params["Dmsq31"]*1E3,
    min=2.3,
    max=2.7,
    step=(2.7-2.3)/100
  )
  dmsq21_slider = widgets.FloatSlider(
    orientation='horizontal',
    description=r"$\Delta\mathrm{m}_{21}^{2}$ [$10^{-5}$ eV]",
    value=osc_params["Dmsq21"]*1E5,
    min=6,
    max=9,
    step=(9-6)/100
  )

  s12sq_slider = widgets.FloatSlider(
    orientation='horizontal',
    description=r"$\mathrm{sin}^{2}(\theta_{12})$",
    value=osc_params["s12sq"],
    min=0.25,
    max=0.36,
    step=(0.36-0.25)/100
  )
  s13sq_slider = widgets.FloatSlider(
    orientation='horizontal',
    description=r"$\mathrm{sin}^{2}(\theta_{13})$",
    value=osc_params["s13sq"],
    min=0.015,
    max=0.025,
    step=(0.025-0.015)/100
  )
  s23sq_slider = widgets.FloatSlider(
    orientation='horizontal',
    description=r"$\mathrm{sin}^{2}(\theta_{23})$",
    value=osc_params["s23sq"],
    min=0.4,
    max=0.7,
    step=(0.7-0.4)/100
  )
  
  delta_slider = widgets.FloatSlider(
    orientation='horizontal',
    description=r"$\delta_\mathrm{CP}/\pi$",
    value=osc_params["delta"]/np.pi,
    min=-1,
    max=1,
    step=(1+1)/100
  )
  
  display(InteractiveOscProbPlot_fig.canvas)
  display(dmsq31_slider)
  display(dmsq21_slider)
  display(s12sq_slider)
  display(s13sq_slider)
  display(s23sq_slider)
  display(delta_slider)
  
  def update(val):
      var_params["Dmsq31"] = dmsq31_slider.value * 1E-3
      var_params["Dmsq21"] = dmsq21_slider.value * 1E-5
      var_params["s12sq"] = s12sq_slider.value
      var_params["s13sq"] = s13sq_slider.value
      var_params["s23sq"] = s23sq_slider.value
      var_params["delta"] = delta_slider.value * np.pi
      
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
  
  dmsq31_slider.observe(update)
  dmsq21_slider.observe(update)
  s12sq_slider.observe(update)
  s13sq_slider.observe(update)
  s23sq_slider.observe(update)
  delta_slider.observe(update)


InteractiveLikelihoodScan_fig = None
def InteractiveLikelihoodScan():
  global InteractiveLikelihoodScan_fig
  if InteractiveLikelihoodScan_fig is not None:
    plt.close(InteractiveLikelihoodScan_fig)

  InteractiveLikelihoodScan_fig = plt.figure(figsize=(6, 4))

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

  axpl = InteractiveLikelihoodScan_fig.add_axes((0.1,0.2,0.39,0.65))

  InteractiveLikelihoodScan_fig.sca(axpl)
  
  drawhist1d(hist=h1, color="#AA3377", lw=2, label=r"Observation")

  osc_params["Dmsq31"] = 2.65e-3

  event_osc_weights_numu_surv = \
          Probability_Matter_LBL(events_nu_numu_cc["E_neutrino"], osc_params, osc_channels=["numu_survival"])
  h2 = hist1d(data=events_nu_numu_cc["E_neutrino"], weights=event_osc_weights_numu_surv*sf, bins=bins)

  numu_surv_pred_l = drawhist1d(hist=h2, color="#4477AA", lw=2, label=r"Model Prediction")
  # bin_centers = (bins[1:] + bins[:-1])/2 
  # plt.errorbar(bin_centers, h2[0], yerr=np.sqrt(h2[0]),  color="#4477AA", lw=0, elinewidth=1)
  
  axpl.set_xlabel(r"$E_{\nu}$ [GeV]", size="large")
  axpl.set_ylabel(r"$P_{\nu_\mu \rightarrow \nu_\mu}$", size="x-large")
  axpl.set_ylim([0,320])
  axpl.legend()
  
  # axpr = InteractiveLikelihoodScan_fig.add_axes((0.6,0.725,0.39,0.26))
  # numu_unosc_evr_lo_r = axpr.fill_between(Es, nu_unosc_evr*0.3*0.9, np.zeros_like(Es), fc="grey", alpha=0.1)
  # nue_surv_prob_lo, = axpr.plot(Es, nue_app_prob, c=colors["nue"], alpha=alpha)
  # nue_surv_prob_l, = axpr.plot(Es, nue_app_prob, c=colors["nue"], label=r"electron neutrino")
  # antinue_surv_prob_lo, = axpr.plot(Es, antinue_app_prob, c=colors["antinue"], alpha=alpha, ls="dashed")
  # antinue_surv_prob_l, = axpr.plot(Es, antinue_app_prob, c=colors["antinue"], ls="dashed", label=r"electron antineutrino")
  # axpr.set_xlabel(r"$E_{\nu}$ [GeV]", size="large")
  # axpr.set_ylabel(r"$P_{\nu_\mu \rightarrow \nu_\mathrm{e}}$", size="large")
  # axpr.set_ylim([0,0.3])
  # axpr.legend()

  axdmsq31 = InteractiveLikelihoodScan_fig.add_axes((0.25, 0.05, 0.5, 0.03))
  dmsq31_slider = Slider(
      ax=axdmsq31,
      label=r"$\Delta\mathrm{m}_{31}^{2}$ [$10^{-3}$ eV]",
      valmin=2.3,
      valmax=2.7,
      valinit=osc_params["Dmsq31"]*1E3,
  )
  
  # The function to be called anytime a slider's value changes
  def update(val):
      var_params["Dmsq31"] = dmsq31_slider.val * 1E-3
      
      # numu_surv_prob = Probability_Matter_LBL(Es, var_params, 
      #                        osc_channels=["numu_survival"])
      # h2 = hist1d(data=events_nu_numu_cc["E_neutrino"], weights=numu_surv_prob*sf, bins=bins)
    
      # numu_surv_pred_l = drawhist1d(hist=h2, color="#4477AA", lw=2, label=r"Model Prediction")
      
      InteractiveOscProbPlot_fig.canvas.draw_idle()
  
  
  # register the update function with each slider
  dmsq31_slider.on_changed(update)
