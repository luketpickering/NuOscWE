import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import Button, FloatSlider, Label, VBox, HBox, Layout, Output
import pandas as pa

from scipy.stats import norm

from .NuFast import *
from .stats import *

from types import SimpleNamespace

IntrLHood = SimpleNamespace()

IntrLHood.binning = np.linspace(start=0.25, stop=8, num=51)
IntrLHood.bin_centers = (IntrLHood.binning[1:] + IntrLHood.binning[:-1])/2 
IntrLHood.colors = SimpleNamespace(
    numu = "#AA3377",
    nue = "#4477AA",
    antinumu = "#EE6677",
    antinue = "#66CCEE",
)
IntrLHood.osc_params = {
    "experimental_baseline_km": 1300,
    "s12sq": 0.31,
    "s13sq": 0.02,
    "s23sq": 0.55,
    "delta": 0.7 * np.pi,
    "Dmsq21": 7.5e-5,
    "Dmsq31": 2.5e-3
  }

IntrLHood.simulated_events_nu_mode = pa.read_csv("simulation/neutrino_mode_events.csv")
IntrLHood.events_nu_numu_cc = IntrLHood.simulated_events_nu_mode[IntrLHood.simulated_events_nu_mode["pid_lepton"] == 13]

def _get_hist(osc_params, exposure_sf, bins):
  return hist1d(data=IntrLHood.events_nu_numu_cc["E_neutrino"], 
                  weights=exposure_sf * Probability_Matter_LBL(IntrLHood.events_nu_numu_cc["E_neutrino"], 
                                                               osc_params, osc_channels=["numu_survival"]), 
                  bins=bins)[0]

def _do_scan(parname, parvals, h_obs, osc_params, exposure_sf, bins):
  vpars = osc_params.copy()
  rtnvals = []
  for x in parvals:
    vpars[parname] = x
    rtnvals.append(Pearson_N2LLH(h_obs, _get_hist(vpars, exposure_sf, bins)))
  return rtnvals

IntrLHood.fig1 = None
def InteractiveLikelihoodScan_1():
  global IntrLHood
  if IntrLHood.fig1 is not None:
    plt.close(IntrLHood.fig1)

  exposure_sf = 0.2
  osc_params = IntrLHood.osc_params.copy()
  
  osc_params["Dmsq31"] = 2.4e-3
  h_obs = _get_hist(osc_params, exposure_sf, bins=IntrLHood.binning)

  plt.ioff()
  IntrLHood.fig1, axes = plt.subplots(nrows=1, ncols=2, \
                                      gridspec_kw=dict(left=0.1,bottom=0.15, top=0.97, wspace=0.3,hspace=0.3), \
                                      figsize=(12,4))
  IntrLHood.fig1.canvas.toolbar_visible = False
  
  axl, axr = axes.flatten()

  obs_l = axl.stairs(h_obs, IntrLHood.binning, color=IntrLHood.colors.numu, lw=2, label=r"Observation")
  
  osc_params["Dmsq31"] = IntrLHood.osc_params["Dmsq31"]

  h_pred = _get_hist(osc_params, exposure_sf, bins=IntrLHood.binning)

  pred_hist_l = axl.stairs(h_pred, IntrLHood.binning, color=IntrLHood.colors.nue, lw=2, label=r"Model Prediction")
  pred_errs_l = axl.errorbar(IntrLHood.bin_centers, h_pred, yerr=np.sqrt(h_pred), color=IntrLHood.colors.nue, lw=0, elinewidth=1)
  
  axl.set_xlabel(r"$E_{\nu}$ [GeV]", size="large")
  axl.set_ylabel(r"Observed Muon Neutrino Events", size="x-large")
  axl.set_ylim([0, np.max(h_pred)*1.25])
  axl.legend()

  dmsq31_range = [2.3, 2.7]
  controls_spec = [
    dict( pname = "Dmsq31", srange = dmsq31_range, sf = 1E3, desc = r"$\Delta\mathrm{m}_{31}^{2}$ [$10^{-3}$ eV]" ),
  ]

  lhood_x = np.linspace(dmsq31_range[0], dmsq31_range[1], 51)
  lhood_y = np.zeros_like(lhood_x)

  lhood_l, = axr.plot(lhood_x, lhood_y, ".", c=IntrLHood.colors.nue)
  lhood_scan_l, = axr.plot([osc_params["Dmsq31"]*1E3, osc_params["Dmsq31"]*1E3],[0,180], c="#FF0000", alpha=0.5)
  
  axr.set_xlabel(controls_spec[0]["desc"], size="large")
  axr.set_ylabel(r"$-2\log(\mathcal{L})$", size="x-large")
  axr.set_ylim([-10, 200])
  
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

  # output = Output(layout={'border': '1px solid black'})
  # controls.append(output)
  
  ncalls = 0
  updating = False
  def update(val):
    nonlocal ncalls, updating
    ncalls = ncalls + 1
    if ncalls < 5:
      return

    if updating:
      return
    updating = True

    var_params = osc_params.copy()
    for i,c in enumerate(controls_spec):
      var_params[c["pname"]] = sliders[i].value / c["sf"]
      
    h_pred = _get_hist(var_params, exposure_sf, bins=IntrLHood.binning)

    nonlocal pred_hist_l, pred_errs_l
    pred_hist_l.remove()
    pred_errs_l.remove()
    
    pred_hist_l = axl.stairs(h_pred, IntrLHood.binning, color=IntrLHood.colors.nue, lw=2, label=r"Model Prediction")
    pred_errs_l = axl.errorbar(IntrLHood.bin_centers, h_pred, yerr=np.sqrt(h_pred),  color=IntrLHood.colors.nue, lw=0, elinewidth=1)

    N2LLH = Pearson_N2LLH(h_obs, h_pred)
    lhood_i = np.argmax(lhood_x > sliders[i].value)
    if sliders[i].value <= lhood_x[0]:
      lhood_i = 0
    if sliders[i].value < lhood_x[-1]:
      lhood_y[lhood_i] = N2LLH

    lhood_l.set_ydata(lhood_y)
    lhood_scan_l.set_xdata([var_params["Dmsq31"]*1E3, var_params["Dmsq31"]*1E3])
    
    IntrLHood.fig1.canvas.draw_idle()

    # output.append_stdout(f"Dmsq31 x10^3 = {var_params['Dmsq31']}, lhood = {N2LLH}\n")
    updating = False

  for s in sliders:
    s.observe(update)
  
  display(IntrLHood.fig1.canvas)
  for c in controls:
    display(c)

IntrLHood.fig2 = None
def InteractiveLikelihoodScan_2():
  global IntrLHood
  if IntrLHood.fig2 is not None:
    plt.close(IntrLHood.fig2)

  exposure_sf = 0.2
  osc_params = IntrLHood.osc_params.copy()

  true_val = 2.4E-3
  osc_params["Dmsq31"] = true_val
  h_obs = _get_hist(osc_params, exposure_sf, bins=IntrLHood.binning)
  h_obs_fluct = poisson_fluctuate(h_obs)

  plt.ioff()
  IntrLHood.fig2, axes = plt.subplots(nrows=1, ncols=3, \
                                      gridspec_kw=dict(left=0.05,bottom=0.15, top=0.97, wspace=0.3, hspace=0.1), \
                                      figsize=(14,4))
  IntrLHood.fig2.canvas.toolbar_visible = False
  
  axl, axr, axrr = axes.flatten()

  obs_l = axl.stairs(h_obs_fluct, IntrLHood.binning, color=IntrLHood.colors.numu, lw=2, label=r"Observation", zorder=1)
  
  osc_params["Dmsq31"] = IntrLHood.osc_params["Dmsq31"]

  h_pred = _get_hist(osc_params, exposure_sf, bins=IntrLHood.binning)

  pred_hist_l = axl.stairs(h_pred, IntrLHood.binning, color=IntrLHood.colors.nue, lw=2, label=r"Model Prediction", zorder=2)
  pred_errs_l = axl.errorbar(IntrLHood.bin_centers, h_pred, yerr=np.sqrt(h_pred), color=IntrLHood.colors.nue, lw=0, elinewidth=1, zorder=2)
  
  axl.set_xlabel(r"$E_{\nu}$ [GeV]", size="large")
  axl.set_ylabel(r"Observed Muon Neutrino Events", size="x-large")
  axl.set_ylim([0, np.max(h_pred)*1.25])
  axl.legend()

  dmsq31_range = [2.3, 2.7]
  controls_spec = [
    dict( pname = "Dmsq31", srange = dmsq31_range, sf = 1E3, desc = r"$\Delta\mathrm{m}_{31}^{2}$ [$10^{-3}$ eV]" ),
  ]

  lhood_x = np.linspace(dmsq31_range[0], dmsq31_range[1], 51)
  lhood_scan = _do_scan("Dmsq31", lhood_x, h_obs_fluct, osc_params, exposure_sf, IntrLHood.binning)
  lhood_y = np.zeros_like(lhood_x)

  lhood_l, = axr.plot(lhood_x, lhood_y, ".", c=IntrLHood.colors.nue)
  lhood_scan_l, = axr.plot([osc_params["Dmsq31"]*1E3, osc_params["Dmsq31"]*1E3],[0,180], c="#FF0000", alpha=0.5)
  
  axr.set_xlabel(controls_spec[0]["desc"], size="large")
  axr.set_ylabel(r"$-2\log(\mathcal{L})$", size="x-large")
  axr.set_ylim([-10, 200])

  deltalhood_ls = [axrr.plot(lhood_x, lhood_y, c=IntrLHood.colors.nue)[0]]
  axrr.set_xlabel(controls_spec[0]["desc"], size="large")
  axrr.set_ylabel(r"$\Delta -2\log(\mathcal{L}) = \Delta \chi^2$", size="x-large")
  axrr.set_ylim([0, 9])

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

  delta_chi2_btn = Button(
      description='Calculate Delta Chi2',
      disabled=False,
      button_style='info', # 'success', 'info', 'warning', 'danger' or ''
      icon='calculator' # (FontAwesome names without the `fa-` prefix)
  )
  controls.append(delta_chi2_btn)

  run_expt_btn = Button(
      description='Run the Experiment',
      disabled=False,
      button_style='info', # 'success', 'info', 'warning', 'danger' or ''
      icon='person-running' # (FontAwesome names without the `fa-` prefix)
  )
  controls.append(run_expt_btn)

  output = Output(layout={'border': '1px solid black'})
  controls.append(output)
  
  ncalls = 0
  updating = False
  def update(val):
    nonlocal ncalls, updating
    ncalls = ncalls + 1
    if ncalls < 5:
      return

    if updating:
      return
    updating = True

    var_params = osc_params.copy()
    for i,c in enumerate(controls_spec):
      var_params[c["pname"]] = sliders[i].value / c["sf"]
      
    h_pred = _get_hist(var_params, exposure_sf, bins=IntrLHood.binning)

    nonlocal pred_hist_l, pred_errs_l
    pred_hist_l.remove()
    pred_errs_l.remove()
    
    pred_hist_l = axl.stairs(h_pred, IntrLHood.binning, color=IntrLHood.colors.nue, lw=2, label=r"Model Prediction",zorder=2)
    pred_errs_l = axl.errorbar(IntrLHood.bin_centers, h_pred, yerr=np.sqrt(h_pred),  color=IntrLHood.colors.nue, lw=0, elinewidth=1,zorder=2)

    N2LLH = Pearson_N2LLH(h_obs_fluct, h_pred)
    lhood_i = np.argmax(lhood_x > sliders[i].value)
    if sliders[i].value <= lhood_x[0]:
      lhood_i = 0
    if sliders[i].value < lhood_x[-1]:
      lhood_y[lhood_i] = N2LLH

    lhood_l.set_ydata(lhood_y)
    lhood_scan_l.set_xdata([var_params["Dmsq31"]*1E3, var_params["Dmsq31"]*1E3])
    
    IntrLHood.fig2.canvas.draw_idle()

    # output.append_stdout(f"Dmsq31 x10^3 = {var_params['Dmsq31']}, lhood = {N2LLH}\n")
    updating = False

  for s in sliders:
    s.observe(update)

  calc_updating = False
  def calc_delta_chi2(val):
    nonlocal calc_updating
    if calc_updating:
      return
    calc_updating = True
    
    xvals = np.linspace(true_val-5E-5,true_val+5E-5,50)
    lhood_scan = _do_scan("Dmsq31", xvals, h_obs_fluct, osc_params, exposure_sf, IntrLHood.binning)
    output.append_stdout(f"{lhood_scan}\n")

    while len(deltalhood_ls) >= 10:
      deltalhood_ls[0].remove()
      deltalhood_ls.pop(0)
    
    for i, l in enumerate(deltalhood_ls):
      l.set_alpha(1 - ((len(deltalhood_ls) - i)*0.1))

    deltalhood_ls.append(axrr.plot(xvals*1E3, lhood_scan - np.min(lhood_scan), c=IntrLHood.colors.nue)[0])
    
    IntrLHood.fig2.canvas.draw_idle()
    calc_updating = False
  
  delta_chi2_btn.on_click(calc_delta_chi2)

  def fluctuate(val):
    nonlocal obs_l, h_obs_fluct, lhood_l, lhood_y

    lhood_y = np.zeros_like(lhood_x)
    lhood_l.remove()
    lhood_l, = axr.plot(lhood_x, lhood_y, ".", c=IntrLHood.colors.nue)
    
    obs_l.remove()
    h_obs_fluct = poisson_fluctuate(h_obs)
    obs_l = axl.stairs(h_obs_fluct, IntrLHood.binning, color=IntrLHood.colors.numu, lw=2, label=r"Observation", zorder=1)

    IntrLHood.fig2.canvas.draw_idle()
  
  run_expt_btn.on_click(fluctuate)
  
  display(IntrLHood.fig2.canvas)
  for c in controls:
    display(c)

IntrLHood.fig3 = None
def InteractiveLikelihoodScan_3():
  pass
