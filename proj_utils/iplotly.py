import numpy as np
import plotly.graph_objects as go
import ipywidgets as wdgt

from proj_utils.NuFast import *

class iPlotlyOsc:
  def __init__(self):
    self.sliders = {}
    self.controls = {}
    self.figs = {}
    self.callbacks = {}

    self.params = {
      "s12sq": 0.31,
      "s13sq": 0.02,
      "s23sq": 0.47,
      "delta": 0.7 * np.pi,
      "Dmsq21": 7.5e-5,
      "Dmsq31": 2.5e-3
    }

    self.control_spec = {
      "s12sq": { "rng": [2.5, 3.6], "sf": 10, "desc":  r"$\mathrm{sin}^{2}(\theta_{12})$ [$10^{-1}$]" },
      "s13sq": { "rng": [1, 2.5], "sf": 1E2, "desc": r"$\mathrm{sin}^{2}(\theta_{13})$ [$10^{-2}$]" },
      "s23sq": { "rng": [4, 7], "sf": 10, "desc": r"$\mathrm{sin}^{2}(\theta_{23})$ [$10^{-1}$]" },
      "Dmsq31": { "rng": [2.3, 2.7], "sf": 1E3, "desc": r"$\Delta\mathrm{m}_{31}^{2}$ [$10^{-3}$ eV]" },
      "Dmsq21": { "rng": [6, 9], "sf": 1E5, "desc": r"$\Delta\mathrm{m}_{21}^{2}$ [$10^{-5}$ eV]" },
      "delta": { "rng": [-1, 1], "sf": 1.0/np.pi, "desc": r"$\delta_\mathrm{CP}/\pi$" },
    }

    self.init_controls()
  
  def controls_update(self, cname, val):
    nv = self.sliders[cname].value / self.control_spec[cname]["sf"]
    print(f"update {cname}: {self.params[cname]} -> {nv}")
    self.params[cname] = nv
    for _,fn in self.callbacks[cname].items():
      fn(val)
  
  def register_callback(self, figname, cnames, fn):
    if isinstance(cnames, list):
      for cn in cnames:
        self.callbacks[cn][figname] = fn
    else:
      self.callbacks[cnames][figname] = fn
  
  def init_controls(self):
    for cname, c in self.control_spec.items():
      self.sliders[cname] = wdgt.FloatSlider(orientation='horizontal',
                                value=self.params[cname]*c["sf"],
                                min=c["rng"][0],
                                max=c["rng"][1],
                                step=(c["rng"][1]-c["rng"][0])/100)
      self.callbacks[cname] = {}     
      self.controls[cname] = wdgt.HBox([wdgt.Label(c["desc"]), self.sliders[cname]])

    # for some pyhton-forsaken reason when I do this in the loop, each slider gets 
    # the last lambda so hey all update delta
    self.sliders["s12sq"].observe(lambda v: self.controls_update("s12sq", v))
    self.sliders["s13sq"].observe(lambda v: self.controls_update("s13sq", v))
    self.sliders["s23sq"].observe(lambda v: self.controls_update("s23sq", v))
    self.sliders["Dmsq31"].observe(lambda v: self.controls_update("Dmsq31", v))
    self.sliders["Dmsq21"].observe(lambda v: self.controls_update("Dmsq21", v))
    self.sliders["delta"].observe(lambda v: self.controls_update("delta", v))

  def numu_survival_prob(self, ERange_GeV, L_km):
    Es = np.linspace(ERange_GeV[0], ERange_GeV[1],1000)
    self.figs["numu_survival_prob"] = go.FigureWidget(
       data=go.Scatter(x=Es, 
                       y=Probability_Matter_LBL(Es, L_km, self.params, osc_channels=["numu_survival"]),
                      mode="lines",
                      name="Survival Probability"),
       layout=go.Layout(template="simple_white", height=480, width=640,
                        yaxis_title='Oscillation Probability',
                        xaxis_title='Neutrino Energy [GeV]',
                        margin=dict(b=10,l=10,t=5,r=5))
    )
    
    def update(val):
      with self.figs["numu_survival_prob"].batch_update():
        self.figs["numu_survival_prob"].data[0].y = Probability_Matter_LBL(Es, L_km, self.params, osc_channels=["numu_survival"])
  
    self.register_callback("numu_survival_prob", "Dmsq31", update)
    return wdgt.VBox([self.figs["numu_survival_prob"], self.controls["Dmsq31"]])