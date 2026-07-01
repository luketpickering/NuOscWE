import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, Label, VBox, HBox, Layout
import pandas as pa

from .NuFast import *
from .stats import *

InteractiveOscProbPlot_fig = None
InteractiveBiProbPlot_fig = None
def InteractiveOscProbPlot(osc_params):
  global InteractiveOscProbPlot_fig, InteractiveBiProbPlot_fig
  if InteractiveOscProbPlot_fig is not None:
    plt.close(InteractiveOscProbPlot_fig)

  if InteractiveBiProbPlot_fig is not None:
    plt.close(InteractiveBiProbPlot_fig)

  colors = {
    "numu": "#AA3377",
    "nue": "#4477AA",
    "antinumu": "#EE6677",
    "antinue": "#66CCEE",
    "E0": "#FEE391",
    "E1": "#FB9A29",
    "E2": "#CC4C02",
  }
  
  Energy_Distribution_GeV = osc_params["experimental_energy_distribution"]

  if "range" in Energy_Distribution_GeV:
    Min_E_GeV, Max_E_GeV = Energy_Distribution_GeV["range"]
    
  else:
    Min_E_GeV = Energy_Distribution_GeV["peak"] - 3 * Energy_Distribution_GeV["width"]
    Min_E_GeV = Min_E_GeV if Min_E_GeV > 0 else Energy_Distribution_GeV["peak"]/5
    Max_E_GeV = Energy_Distribution_GeV["peak"] + 3 * Energy_Distribution_GeV["width"]

  EPeak = osc_params["experimental_energy_distribution"]["peak"]
  EWidth = osc_params["experimental_energy_distribution"]["width"]

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

  InteractiveBiProbPlot_fig, bpax = plt.subplots(figsize=(5,5))
  InteractiveBiProbPlot_fig.canvas.toolbar_visible = False
  
  axpl, axpr, axel, axer = axes.flatten()
  
  alpha = 0.3

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

  axpr.plot([EPeak-EWidth,EPeak-EWidth],[0, 0.125], c=colors["E0"], alpha=0.5)
  axpr.plot([EPeak,EPeak],[0, 0.125], c=colors["E1"], alpha=0.5)
  axpr.plot([EPeak+EWidth,EPeak+EWidth],[0, 0.125], c=colors["E2"], alpha=0.5)

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

  biprob_dcp_vals = np.linspace(-np.pi, np.pi, 25)
  def biprob(E, osc_params):
    nue_app_biprob = []
    antinue_app_biprob = []
    vop = osc_params.copy()

    onue_app_prob, oantinue_app_prob = Probability_Matter_LBL(E, vop, 
                                                              osc_channels=["nue_appearance", "antinue_appearance"])
    
    for v in biprob_dcp_vals:
      vop["delta"] = v
      nue_app_prob, antinue_app_prob = Probability_Matter_LBL(E, vop, 
                                                              osc_channels=["nue_appearance", "antinue_appearance"])
      nue_app_biprob.append(nue_app_prob)
      antinue_app_biprob.append(antinue_app_prob)
    return ([onue_app_prob], [oantinue_app_prob], nue_app_biprob, antinue_app_biprob)

  onue_app_prob, oantinue_app_prob, nue_app_biprob, antinue_app_biprob = biprob(EPeak-EWidth,var_params)
  bp_ovalE0_lo, = bpax.plot(nue_app_biprob, antinue_app_biprob, c=colors["E0"], alpha=alpha)
  bp_ovalE0_l, = bpax.plot(nue_app_biprob, antinue_app_biprob, c=colors["E0"], label=f"{EPeak-EWidth:.2} GeV")
  bp_ovalE0_po, = bpax.plot(onue_app_prob, oantinue_app_prob, "o", c="#FF0000", alpha=alpha)
  bp_ovalE0_p, = bpax.plot(onue_app_prob, oantinue_app_prob, "o", c="#FF0000")

  onue_app_prob, oantinue_app_prob, nue_app_biprob, antinue_app_biprob = biprob(EPeak,var_params)
  bp_ovalE1_lo, = bpax.plot(nue_app_biprob, antinue_app_biprob, c=colors["E1"], alpha=alpha)
  bp_ovalE1_l, = bpax.plot(nue_app_biprob, antinue_app_biprob, c=colors["E1"], label=f"{EPeak:.2} GeV")
  bp_ovalE1_po, = bpax.plot(onue_app_prob, oantinue_app_prob, "o", c="#FF0000", alpha=alpha)
  bp_ovalE1_p, = bpax.plot(onue_app_prob, oantinue_app_prob, "o", c="#FF0000")

  onue_app_prob, oantinue_app_prob, nue_app_biprob, antinue_app_biprob = biprob(EPeak+EWidth,var_params)
  bp_ovalE2_lo, = bpax.plot(nue_app_biprob, antinue_app_biprob, c=colors["E2"], alpha=alpha)
  bp_ovalE2_l, = bpax.plot(nue_app_biprob, antinue_app_biprob, c=colors["E2"], label=f"{EPeak+EWidth:.2} GeV")
  bp_ovalE2_po, = bpax.plot(onue_app_prob, oantinue_app_prob, "o", c="#FF0000", alpha=alpha)
  bp_ovalE2_p, = bpax.plot(onue_app_prob, oantinue_app_prob, "o", c="#FF0000",label=r"Current $\delta_\mathrm{CP}$ Value")
  
  bpax.set_xlabel(f"Electron Neutrino Appearance Probability")
  bpax.set_ylabel(f"Electron Antieutrino Appearance Probability")
  bpax.set_ylim([0,0.125])
  bpax.set_xlim([0,0.125])
  bpax.legend()
  
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

    onue_app_prob, oantinue_app_prob, nue_app_biprob, antinue_app_biprob = biprob(EPeak-EWidth,var_params)
    bp_ovalE0_l.set_xdata(nue_app_biprob)
    bp_ovalE0_l.set_ydata(antinue_app_biprob)
    bp_ovalE0_p.set_xdata(onue_app_prob)
    bp_ovalE0_p.set_ydata(oantinue_app_prob)

    onue_app_prob, oantinue_app_prob, nue_app_biprob, antinue_app_biprob = biprob(EPeak,var_params)
    bp_ovalE1_l.set_xdata(nue_app_biprob)
    bp_ovalE1_l.set_ydata(antinue_app_biprob)
    bp_ovalE1_p.set_xdata(onue_app_prob)
    bp_ovalE1_p.set_ydata(oantinue_app_prob)

    onue_app_prob, oantinue_app_prob, nue_app_biprob, antinue_app_biprob = biprob(EPeak+EWidth,var_params)
    bp_ovalE2_l.set_xdata(nue_app_biprob)
    bp_ovalE2_l.set_ydata(antinue_app_biprob)
    bp_ovalE2_p.set_xdata(onue_app_prob)
    bp_ovalE2_p.set_ydata(oantinue_app_prob)
    
    InteractiveOscProbPlot_fig.canvas.draw_idle()
    InteractiveBiProbPlot_fig.canvas.draw_idle()

  for s in sliders:
    s.observe(update)

  plot_container = HBox([InteractiveOscProbPlot_fig.canvas,InteractiveBiProbPlot_fig.canvas])
  control_container = HBox([
    VBox([controls[i] for i in range(3)],layout=Layout(align_items="flex-end")),
    VBox([controls[i+3] for i in range(3)],layout=Layout(align_items="flex-end")),
  ],layout=Layout(justify_content="flex-start"))
  
  display(plot_container)
  display(control_container)
