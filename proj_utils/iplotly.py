import numpy as np
import plotly.graph_objects as go
import ipywidgets as wdgt

from proj_utils.NuFast import *
from proj_utils.stats import gaus
from proj_utils.xsec import NumuCCTotInterp, NumubCCTotInterp
from proj_utils.flux import DUNENDNumu, NOvANDNumu, SKNumu

class iPlotlyOsc:
  def __init__(self):
    self.sliders = {}
    self.labels = {}
    self.figs = {}
    self.callbacks = {}

    self.params = {
      "s12sq": 0.31,
      "s13sq": 0.02,
      "s23sq": 0.47,
      "delta": 0.7 * np.pi,
      "Dmsq21": 7.5e-5,
      "Dmsq31": 2.5e-3,
      
      "gaussbeam_mu": 1,
      "gaussbeam_w": 0.5,
      "expt_baseline": 500,
      "expt_mass": 20
    }

    self.lines = {
      "numu": dict(color="#AA3377"),
      "nue": dict(color="#4477AA"),
      "antinumu": dict(color="#EE6677"),
      "antinue": dict(color="#66CCEE"),
    }

    self.control_spec = {
      "s13sq": { "rng": [1, 2.5], "nd":100, "sf": 1E2, "desc": r"$\mathrm{sin}^{2}({\color{red}\theta_{13}})$ [$10^{-2}$]" },
      "s23sq": { "rng": [4, 7], "nd":100, "sf": 10, "desc": r"$\mathrm{sin}^{2}({\color{blue}\theta_{23}})$ [$10^{-1}$]" },
      "Dmsq32": { "rng": [2.3, 2.7], "nd":100, "sf": 1E3, "desc": r"${\color{purple}\Delta{}m^2_{32}}$ [$10^{-3}$ eV]" },
      "Dmsq21": { "rng": [6, 9], "nd":100, "sf": 1E5, "desc": r"${\color{cyan}\Delta{}m^2_{21}}$ [$10^{-5}$ eV]" },
      "delta": { "rng": [-1, 1], "nd":100, "sf": 1.0/np.pi, "desc": r"${\color{green}\delta_\mathrm{CP}}/\pi$" },

      "gaussbeam_mu": { "rng": [0.1, 7], "nd":100, "sf": 1.0, "desc": r"Beam Mean Energy [GeV]" },
      "gaussbeam_w": { "rng": [0.1, 2], "nd":100, "sf": 1.0, "desc": "Beam Spectral Width [GeV]" },
      "expt_baseline": { "rng": [0, 5000], "nd":1000, "sf": 1.0, "desc": "Baseline [km]" },
      "expt_mass": { "rng": [0, 250], "nd":250, "sf": 1.0, "desc": "Detector mass [kt]" },
    }

    self.init_controls()

  def set_param(self, cname, val):
    if cname == "Dmsq32":
      self.params["Dmsq31"] = val + self.params["Dmsq21"]
    elif cname not in self.params:
      return
    else:
      self.params[cname] = val
  
  def get_param(self, cname):
    if cname == "Dmsq32":
      return self.params["Dmsq31"] - self.params["Dmsq21"]
    elif cname not in self.params:
      return None
    else:
      return self.params[cname]
  
  def controls_update(self, cname, val):
    self.set_param(cname, self.sliders[cname].value / self.control_spec[cname]["sf"])
    for _,fn in self.callbacks[cname].items():
      fn(val)
  
  def register_callback(self, figname, cnames, fn):
    if isinstance(cnames, list):
      for cn in cnames:
        self.callbacks[cn][figname] = fn
    else:
      self.callbacks[cnames][figname] = fn

  def get_control_box(self, controls):
    return wdgt.HBox([
      wdgt.VBox([self.labels[c] for c in controls]),
      wdgt.VBox([self.sliders[c] for c in controls]),
    ])
  
  def init_controls(self):
    for cname, c in self.control_spec.items():
      self.sliders[cname] = wdgt.FloatSlider(orientation='horizontal',
                                value=self.get_param(cname)*c["sf"],
                                min=c["rng"][0],
                                max=c["rng"][1],
                                step=(c["rng"][1]-c["rng"][0])/c["nd"])
      self.callbacks[cname] = {}     
      self.labels[cname] = wdgt.Label(c["desc"])

    # for some python-forsaken reason when I do this in the loop, each slider gets 
    # the last lambda so hey all update delta
    self.sliders["s13sq"].observe(lambda v: self.controls_update("s13sq", v))
    self.sliders["s23sq"].observe(lambda v: self.controls_update("s23sq", v))
    self.sliders["Dmsq32"].observe(lambda v: self.controls_update("Dmsq32", v))
    self.sliders["Dmsq21"].observe(lambda v: self.controls_update("Dmsq21", v))
    self.sliders["delta"].observe(lambda v: self.controls_update("delta", v))
    self.sliders["gaussbeam_mu"].observe(lambda v: self.controls_update("gaussbeam_mu", v))
    self.sliders["gaussbeam_w"].observe(lambda v: self.controls_update("gaussbeam_w", v))
    self.sliders["expt_baseline"].observe(lambda v: self.controls_update("expt_baseline", v))
    self.sliders["expt_mass"].observe(lambda v: self.controls_update("expt_mass", v))
    
  def numu_survival_prob_loe(self):
    Ls = np.linspace(0.1, 2*np.pi/self.params["Dmsq31"], 1000)
    Es = np.ones_like(Ls)
    self.figs["numu_survival_prob_loe"] = go.FigureWidget(
       data=go.Scatter(x=Ls/Es, 
                       y=Probability_Matter_LBL(Es, Ls, self.params, osc_channels=["numu_survival"]),
                       mode="lines",
                       line=self.lines["numu"],
                       name="Survival Probability"),
       layout=go.Layout(template="simple_white", height=480, width=640,
                        yaxis_title='Oscillation Probability',
                        xaxis_title='L/E [km/GeV]',
                        margin=dict(b=10,l=10,t=5,r=5))
    )
    self.figs["numu_survival_prob_loe"].update_yaxes(range=[0,1.1])
    
    def update(val):
      with self.figs["numu_survival_prob_loe"].batch_update():
        self.figs["numu_survival_prob_loe"].data[0].y = Probability_Matter_LBL(Es, Ls, self.params, osc_channels=["numu_survival"])

    controls = ["s13sq", "s23sq", "Dmsq32"]
    self.register_callback("numu_survival_prob_loe", controls, update)
    return wdgt.HBox([self.figs["numu_survival_prob_loe"], self.get_control_box(controls)])

  def numu_survival_prob(self, ERange_GeV, L_km):
    Es = np.linspace(ERange_GeV[0], ERange_GeV[1],1000)
    self.figs["numu_survival_prob"] = go.FigureWidget(
       data=go.Scatter(x=Es, 
                       y=Probability_Matter_LBL(Es, L_km, self.params, osc_channels=["numu_survival"]),
                       mode="lines",
                       line=self.lines["numu"],
                       name="Survival Probability"),
       layout=go.Layout(template="simple_white", height=480, width=640,
                        yaxis_title='Survival Probability',
                        xaxis_title='Neutrino Energy [GeV]',
                        margin=dict(b=10,l=10,t=5,r=5))
    )
    self.figs["numu_survival_prob"].update_yaxes(range=[0, 1.1])
    
    def update(val):
      with self.figs["numu_survival_prob"].batch_update():
        self.figs["numu_survival_prob"].data[0].y = Probability_Matter_LBL(Es, L_km, self.params, osc_channels=["numu_survival"])
  
    controls = ["s13sq", "s23sq", "Dmsq32"]
    self.register_callback("numu_survival_prob", controls, update)
    return wdgt.HBox([self.figs["numu_survival_prob"], self.get_control_box(controls)])

  def xsecs(self):
    Exsec = np.linspace(0.05, 30, 100)
    self.figs["xsecs"] = go.FigureWidget(
       data=[ go.Scatter(x=Exsec, 
                         y=NumuCCTotInterp(Exsec),
                         mode="lines",
                         line=self.lines["numu"],
                         name="Neutrino"),
              go.Scatter(x=Exsec, 
                         y=NumubCCTotInterp(Exsec),
                         mode="lines",
                         line=self.lines["antinumu"],
                         name="Antineutrino")
            ],
       layout=go.Layout(template="simple_white", height=480, width=640,
                        yaxis_title='Charged-Current neutrino-Ar Cross Section [pb/Nucleon]',
                        xaxis_title='Neutrino Energy [GeV]',
                        margin=dict(b=10,l=10,t=5,r=5),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ))
    )
    
    return self.figs["xsecs"]

  def gen_beam(self, beam_type):
    if beam_type == "Gaussian":
      mu, w = self.params["gaussbeam_mu"], self.params["gaussbeam_w"]
      def gbeam(E):
        return gaus(E, mu, w)/gaus(mu, mu, w)
      return gbeam
    if beam_type == "DUNE":
      return DUNENDNumu
    if beam_type == "NOvA":
      return NOvANDNumu
    if beam_type == "T2K":
      return SKNumu
    
  def evrate(self):
    Es = np.linspace(0.01, 8, 1500)

    arb_sf = 1E3
    
    beam = self.gen_beam("Gaussian")
    m_kt = self.params["expt_mass"]
    numu_rate = beam(Es) * NumuCCTotInterp(Es) * arb_sf * m_kt
    numub_rate = beam(Es) * NumubCCTotInterp(Es) * arb_sf * m_kt
    numup, numubp, nuep, nuebp = Probability_Matter_LBL(Es, self.params["expt_baseline"], self.params, 
                                         osc_channels=["numu_survival", "antinumu_survival", "nue_appearance", "antinue_appearance"])

    self.figs["evrate"] = go.FigureWidget(
       data=[ 
             go.Scatter(x=Es, 
                         y=numu_rate,
                         mode="lines",
                         line_color="black",
                         name="No Osc."),
              go.Scatter(x=Es, 
                         y=numup * numu_rate,
                         mode="lines",
                         line=self.lines["numu"],
                         name="Surv. NuMu"),
              go.Scatter(x=Es, 
                         y=nuep * numu_rate,
                         mode="lines",
                         line=self.lines["nue"],
                         name="App. NuE")
            ],
       layout=go.Layout(template="simple_white", height=240, width=640,
                        yaxis_title='Neutrino Events',
                        xaxis_title='Neutrino Energy [GeV]',
                        margin=dict(b=10,l=10,t=5,r=10),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ))
    )
    self.figs["evrate"].update_yaxes(exponentformat="power", range=[0,np.max(numu_rate)*1.2])

    self.figs["antievrate"] = go.FigureWidget(
       data=[ go.Scatter(x=Es, 
                         y=numub_rate,
                         mode="lines",
                         line_color="black",
                         name="No Osc."),
              go.Scatter(x=Es, 
                         y=numub_rate * numubp,
                         mode="lines",
                         line=self.lines["antinumu"],
                         name="Surv. NuMuBar"),
              go.Scatter(x=Es, 
                         y=numub_rate * nuebp,
                         mode="lines",
                         line=self.lines["antinue"],
                         name="App. NuEBar")
            ],
       layout=go.Layout(template="simple_white", height=240, width=640,
                        yaxis_title='Antineutrino Events',
                        xaxis_title='Neutrino Energy [GeV]',
                        margin=dict(b=10,l=10,t=5,r=10),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ))
    )
    self.figs["antievrate"].update_yaxes(exponentformat="power", range=[0,np.max(numub_rate)*1.2])

    beam_selector = wdgt.ToggleButtons(
      options=['Gaussian', 'DUNE', 'NOvA', 'T2K'],
      description='Choose Beam:')

    fixy = wdgt.ToggleButton(
      value=False,
      description='Limit y Range',
      layout=wdgt.Layout(width="200px"))
    
    show_unosc = wdgt.ToggleButton(
      value=True,
      description='Show No Oscillations',
      layout=wdgt.Layout(width="200px"))
    
    show_surv = wdgt.ToggleButton(
      value=True,
      description='Show Muon Neutrino Survival',
      layout=wdgt.Layout(width="200px"))
    
    incl_oors = wdgt.ToggleButton(
      value=False,
      description='Enable Beam Dispersion',
      layout=wdgt.Layout(width="200px"))
    
    def update(val):

      L = self.params["expt_baseline"]
      oors = 1 if not incl_oors.value else 10*(500*500/max(0.5*0.5, L*L)) # treat all NDs as 1km away

      beam = self.gen_beam(beam_selector.value)
      m_kt = self.params["expt_mass"]
      numu_rate = beam(Es) * NumuCCTotInterp(Es) * oors * arb_sf * m_kt
      numub_rate = beam(Es) * NumubCCTotInterp(Es) * oors * arb_sf * m_kt
      numup, numubp, nuep, nuebp = Probability_Matter_LBL(Es, L, self.params, 
                                       osc_channels=["numu_survival", "antinumu_survival", "nue_appearance", "antinue_appearance"])
      
      with self.figs["evrate"].batch_update():
        self.figs["evrate"].update_traces(selector=0, patch=dict(visible=True if show_unosc.value else "legendonly"))
        if show_unosc.value:
          self.figs["evrate"].data[0].y = numu_rate
        self.figs["evrate"].update_traces(selector=1, patch=dict(visible=True if show_surv.value else "legendonly"))
        if show_surv.value:
          self.figs["evrate"].data[1].y = numup * numu_rate
        self.figs["evrate"].data[2].y = nuep * numu_rate

        if show_unosc.value:
          ymax = np.max(self.figs["evrate"].data[0].y)
          fixedrate = 600
        elif show_surv.value:
          ymax = np.max(self.figs["evrate"].data[1].y)
          fixedrate = 250
        else:
          ymax = np.max(self.figs["evrate"].data[2].y)
          fixedrate = 55
        if fixy.value:
          self.figs["evrate"].update_yaxes(range=[0, fixedrate])
        else:
          self.figs["evrate"].update_yaxes(range=[0, ymax*1.2])

      with self.figs["antievrate"].batch_update():
        self.figs["antievrate"].update_traces(selector=0, patch=dict(visible=True if show_unosc.value else "legendonly"))
        if show_unosc.value:
          self.figs["antievrate"].data[0].y = numub_rate
        self.figs["antievrate"].update_traces(selector=1, patch=dict(visible=True if show_surv.value else "legendonly"))
        if show_surv.value:
          self.figs["antievrate"].data[1].y = numubp * numub_rate
        self.figs["antievrate"].data[2].y = nuebp * numub_rate

        if show_unosc.value:
          ymax = np.max(self.figs["antievrate"].data[0].y)
        elif show_surv.value:
          ymax = np.max(self.figs["antievrate"].data[1].y)
        else:
          ymax = np.max(self.figs["antievrate"].data[2].y)
        if fixy.value:
          self.figs["antievrate"].update_yaxes(range=[0, fixedrate])
        else:
          self.figs["antievrate"].update_yaxes(range=[0, ymax*1.2])
    
    beam_selector.observe(update)
    fixy.observe(update)
    show_unosc.observe(update)
    show_surv.observe(update)
    incl_oors.observe(update)

    toggle_btns = wdgt.VBox([ wdgt.Label("Visualisation Controls:"),
                              wdgt.HBox([fixy, incl_oors], layout=wdgt.Layout(margin_top="2em")), 
                              wdgt.HBox([show_unosc, show_surv], layout=wdgt.Layout(margin_top="2em"))],
                            layout=wdgt.Layout(margin_top="2em"))
    
    controls = ["s13sq", "s23sq", "Dmsq32", "Dmsq21", "delta", "gaussbeam_mu", "gaussbeam_w", "expt_baseline", "expt_mass"]
    self.register_callback("evrate", controls, update)
    return wdgt.HBox([ wdgt.VBox([self.figs["evrate"], self.figs["antievrate"]]), wdgt.VBox([self.get_control_box(controls), beam_selector, toggle_btns]) ])

  def bievent(self):

    def cdcp(beam, Es):
      odcp = self.params["delta"]
      nuer = beam(Es) * NumuCCTotInterp(Es)
      nuebr = beam(Es) * NumubCCTotInterp(Es)
      nue = np.zeros(51)
      nueb = np.zeros(51)
      
      for i in range(51):
        self.params["delta"] = -np.pi + 2*np.pi*(i/50)
        nuep, nuebp = Probability_Matter_LBL(Es, self.params["expt_baseline"], self.params, 
                                             osc_channels=["nue_appearance", "antinue_appearance"])
        nue[i] = np.sum(nuer * nuep)
        nueb[i] = np.sum(nuebr * nuebp)
      self.params["delta"] = odcp

      return nue, nueb

    beam = self.gen_beam("DUNE")

    nue0, nueb0 = cdcp(beam, np.linspace(0.25,1,50))
    nue1, nueb1 = cdcp(beam, np.linspace(1,2,50))
    nue2, nueb2 = cdcp(beam, np.linspace(2,6,50))
    
    self.figs["bievent"] = go.FigureWidget(
       data=[ 
             go.Scatter(x=nue0, 
                         y=nueb0,
                         mode="lines",
                         name="Before Oscillations"),
             go.Scatter(x=nue1, 
                         y=nueb1,
                         mode="lines",
                         name="Before Oscillations"),
             go.Scatter(x=nue2, 
                         y=nueb2,
                         mode="lines",
                         name="Before Oscillations")
            ],
       layout=go.Layout(template="simple_white", height=240, width=640,
                        yaxis_title='Neutrino Events',
                        xaxis_title='Neutrino Energy [GeV]',
                        margin=dict(b=10,l=10,t=5,r=5))
    )
    return self.figs["bievent"]

  def nue_appearance_prob(self, L_km):
    Es = np.linspace(0.05*L_km * self.params["Dmsq31"]/2*np.pi, 1.5*L_km * self.params["Dmsq31"]/2*np.pi,1000)
    nuep, nuebp = Probability_Matter_LBL(Es, L_km, self.params, osc_channels=["nue_appearance", "antinue_appearance"])
    self.figs["nue_appearance_prob"] = go.FigureWidget(
       data=[ go.Scatter(x=Es, 
                         y=nuep,
                         mode="lines",
                         line=self.lines["nue"],
                         name="Electron Neutrinos"),
              go.Scatter(x=Es, 
                         y=nuebp,
                         mode="lines",
                         line=self.lines["antinue"],
                         name="Electron Antineutrinos")
            ],
       layout=go.Layout(template="simple_white", height=480, width=640,
                        yaxis_title='Appearance Probability',
                        xaxis_title='Neutrino Energy [GeV]',
                        margin=dict(b=10,l=10,t=5,r=5),
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                       ))
    )
    self.figs["nue_appearance_prob"].update_yaxes(range=[0, 0.2])
    
    def update(val):
      with self.figs["nue_appearance_prob"].batch_update():
        nuep, nuebp = Probability_Matter_LBL(Es, L_km, self.params, osc_channels=["nue_appearance", "antinue_appearance"])
        self.figs["nue_appearance_prob"].data[0].y = nuep
        self.figs["nue_appearance_prob"].data[1].y = nuebp
  
    controls = ["s13sq", "s23sq", "Dmsq32", "Dmsq21", "delta"]
    self.register_callback("nue_appearance_prob", controls, update)
    return wdgt.HBox([self.figs["nue_appearance_prob"], self.get_control_box(controls)])