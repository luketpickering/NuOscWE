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
      "expt_mass": 20,

      "use_vacuum_prob": False
    }

    self.ecols = ["#EE7733","#EE3377","#AA4499"]
    self.lines = {
      "numu": dict(color="#AA3377"),
      "nue": dict(color="#4477AA"),
      "antinumu": dict(color="#EE6677"),
      "antinue": dict(color="#66CCEE"),
      
      "E0": dict(color=self.ecols[0]),
      "E1": dict(color=self.ecols[1]),
      "E2": dict(color=self.ecols[2]),
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
      options=['Gaussian', 'DUNE', 'NOvA', 'T2K'])

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
    
    osc_controls = ["s13sq", "s23sq", "Dmsq32", "Dmsq21", "delta"]
    self.register_callback("evrate", osc_controls, update)
    det_controls = ["expt_baseline", "expt_mass"]
    self.register_callback("evrate", det_controls, update)
    beam_controls = ["gaussbeam_mu", "gaussbeam_w"]
    self.register_callback("evrate", beam_controls, update)
    return wdgt.HBox([ wdgt.VBox([self.figs["evrate"], self.figs["antievrate"]]), 
                       wdgt.VBox([wdgt.Label("Oscillation Parameters:", style=dict(font_weight="Bold", font_size="14pt")), self.get_control_box(osc_controls),
                                  wdgt.Label("Select Beam Type:", style=dict(font_weight="Bold", font_size="14pt")), beam_selector,
                                  wdgt.Label("Beam Controls:", style=dict(font_weight="Bold", font_size="14pt")), self.get_control_box(beam_controls), incl_oors,
                                  wdgt.Label("Detector Controls:", style=dict(font_weight="Bold", font_size="14pt")), self.get_control_box(det_controls),
                                  wdgt.Label("Visualisation Controls:", style=dict(font_weight="Bold", font_size="14pt")),
                                  wdgt.HBox([fixy, show_unosc, show_surv]) ]) ])

  def bievent(self):

    novpts = 40
    arb_sf = 5E4

    def get_oval_points(Es, beam, do_integral=True):
      odcp = self.params["delta"]
      nue_rate = beam(Es) * NumuCCTotInterp(Es) * arb_sf
      antinue_rate = beam(Es) * NumubCCTotInterp(Es) * arb_sf
      L = self.params["expt_baseline"]

      if do_integral:
        obs = np.zeros((2,novpts + 1))
        curr_obs = np.zeros(2)
          
        for i in range(novpts + 2):
          if i == (novpts+1):
            self.params["delta"] = odcp
          else:
            self.params["delta"] = -np.pi + 2*np.pi*(i/float(novpts))
            
          nuep, nuebp = Probability_Matter_LBL(Es, L, self.params, 
                                               osc_channels=["nue_appearance", "antinue_appearance"])
          num_nue = np.sum(nue_rate * nuep)
          num_antinue = np.sum(antinue_rate * nuebp)

          if i == (novpts+1):
            curr_obs[0] = num_nue
            curr_obs[1] = num_antinue
            break
          
          obs[0,i] = num_nue
          obs[1,i] = num_antinue
      else:
        obs = np.zeros((2, novpts + 1, Es.shape[0]))
        curr_obs = np.zeros((2, Es.shape[0]))
          
        for i in range(novpts + 2):
          if i == (novpts+1):
            self.params["delta"] = odcp
          else:
            self.params["delta"] = -np.pi + 2*np.pi*(i/float(novpts))

          nuep, nuebp = Probability_Matter_LBL(Es, L, self.params, 
                                               osc_channels=["nue_appearance", "antinue_appearance"])

          num_nue = nue_rate * nuep
          num_antinue = antinue_rate * nuebp

          if i == (novpts+1):
            curr_obs[0, :] = num_nue
            curr_obs[1, :] = num_antinue
            break
          
          obs[0,i,:] = num_nue
          obs[1,i,:] = num_antinue

      return obs, curr_obs
        

    beam = self.gen_beam("Gaussian")
    Es = np.linspace(0.01, 8, 500)
    (obs_nue, obs_antinue), (curr_nue, curr_antinue) = get_oval_points(Es, beam)
   
    rEs = np.linspace(0.01, 8, 1500)
    numu_rate = beam(rEs) * NumuCCTotInterp(rEs) * arb_sf
    numub_rate = beam(rEs) * NumubCCTotInterp(rEs) * arb_sf
    nuep, nuebp = Probability_Matter_LBL(rEs, self.params["expt_baseline"], self.params, osc_channels=["nue_appearance", "antinue_appearance"])

    self.figs["bievrate"] = go.FigureWidget(
       data=[ 
              go.Scatter(x=rEs, 
                         y=nuep * numu_rate,
                         mode="lines",
                         line=self.lines["nue"],
                         name="App. NuE"),
              go.Scatter(x=rEs, 
                         y=nuebp * numub_rate,
                         mode="lines",
                         line=self.lines["antinue"],
                         name="App. NuEBar"),
              go.Scatter(x=[0,0], 
                         y=[0,0],
                         mode="lines",
                         line=self.lines["E0"],
                         showlegend=False,
                         visible=False),
              go.Scatter(x=[0,0], 
                         y=[0,0],
                         mode="lines",
                         line=self.lines["E1"],
                         showlegend=False,
                         visible=False),
              go.Scatter(x=[0,0], 
                         y=[0,0],
                         mode="lines",
                         line=self.lines["E2"],
                         showlegend=False,
                         visible=False)
            ],
       layout=go.Layout(template="simple_white", height=240, width=480,
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

    self.figs["bievent"] = go.FigureWidget(
       data=[ 
             go.Scatter(x=obs_nue, 
                        y=obs_antinue,
                        mode="lines",
                        name="Total NuE App. Rate",
                        line=self.lines["E0"],
                       ),
         
             go.Scatter(x=[curr_nue,], 
                        y=[curr_antinue,],
                        mode="markers",
                        marker=dict(symbol=["star"], color="black", size=8, line_color="black", line_width=1),
                        name="Current dcp Value",
                       ),
         
             go.Scatter(x=obs_nue, 
                        y=obs_antinue,
                        mode="lines",
                        line=self.lines["E1"],
                        showlegend=False,
                        visible=False,
                       ),
         
             go.Scatter(x=[curr_nue,], 
                        y=[curr_antinue,],
                        mode="markers",
                        marker=dict(symbol=["star"], color="black", size=8, line_color="black", line_width=1),
                        showlegend=False,
                        visible=False
                       ),
                        
              go.Scatter( x=obs_nue, 
                        y=obs_antinue,
                        mode="lines",
                        line=self.lines["E2"],
                        showlegend=False,
                        visible=False
                       ),
         
             go.Scatter(x=[curr_nue,], 
                        y=[curr_antinue,],
                        mode="markers",
                        marker=dict(symbol=["star"], color="black", size=8, line_color="black", line_width=1),
                        showlegend=False,
                        visible=False
                       ),
            ],
       layout=go.Layout(template="simple_white", height=480, width=480,
                        yaxis_title='Electron Antineutrino Events',
                        xaxis_title='Electron Neutrino Events',
                        margin=dict(b=10,l=10,t=5,r=5),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ))
    )
    self.figs["bievent"].update_xaxes(range=[0, np.max(obs_nue)*1.2])
    self.figs["bievent"].update_yaxes(range=[0, np.max(obs_antinue)*1.2])
    
    beam_selector = wdgt.ToggleButtons(
      options=['Gaussian', 'DUNE', 'NOvA', 'T2K'],
      description='Choose Beam:')

    dointeg_btn = wdgt.ToggleButtons(
      options=['Total', 'Single Energies'],
      description='Choose Event Rate Type:')

    energy_sliders = [ wdgt.FloatSlider(orientation='horizontal',
                                         value=x,
                                         min=0.1,
                                         max=7.9,
                                         step=0.025) for x in range(3) ]
    
    def update(v):    
      dointeg = dointeg_btn.value == "Total"
      (obs_nue, obs_antinue), (curr_nue, curr_antinue) = get_oval_points(Es if dointeg else np.array([energy_sliders[i].value for i in range(3)]), 
                                                  beam, dointeg)

      with self.figs["bievrate"].batch_update():
        numu_rate = beam(rEs) * NumuCCTotInterp(rEs) * arb_sf
        numub_rate = beam(rEs) * NumubCCTotInterp(rEs) * arb_sf
        nuep, nuebp = Probability_Matter_LBL(rEs, self.params["expt_baseline"], self.params, osc_channels=["nue_appearance", "antinue_appearance"])
        
        self.figs["bievrate"].data[0].y = nuep * numu_rate
        self.figs["bievrate"].data[1].y = nuebp * numub_rate
        
        if dointeg:
          self.figs["bievrate"].update_traces(selector=2, patch=dict(visible=False))
          self.figs["bievrate"].update_traces(selector=3, patch=dict(visible=False))
          self.figs["bievrate"].update_traces(selector=4, patch=dict(visible=False))
        else:
          ymax = np.max(self.figs["bievrate"].data[0].y)*1.1
          self.figs["bievrate"].update_traces(selector=2, patch=dict(visible=True))
          self.figs["bievrate"].data[2].x = [energy_sliders[0].value,energy_sliders[0].value]
          self.figs["bievrate"].data[2].y = [0, ymax]
          self.figs["bievrate"].update_traces(selector=3, patch=dict(visible=True))
          self.figs["bievrate"].data[3].x = [energy_sliders[1].value,energy_sliders[1].value]
          self.figs["bievrate"].data[3].y = [0, ymax]
          self.figs["bievrate"].update_traces(selector=4, patch=dict(visible=True))
          self.figs["bievrate"].data[4].x = [energy_sliders[2].value,energy_sliders[2].value]
          self.figs["bievrate"].data[4].y = [0, ymax]
      
      with self.figs["bievent"].batch_update():
        if dointeg:
          self.figs["bievent"].update_traces(selector=0, patch=dict(name="Total NuE(Bar) App. Rate"))
          self.figs["bievent"].data[0].x = obs_nue
          self.figs["bievent"].data[0].y = obs_antinue
          self.figs["bievent"].data[1].x = [curr_nue,]
          self.figs["bievent"].data[1].y = [curr_antinue,]
          
          self.figs["bievent"].update_traces(selector=2, patch=dict(visible=False))
          self.figs["bievent"].update_traces(selector=3, patch=dict(visible=False))
          self.figs["bievent"].update_traces(selector=4, patch=dict(visible=False))
          self.figs["bievent"].update_traces(selector=5, patch=dict(visible=False))

          self.figs["bievent"].update_xaxes(range=[0, np.max(obs_nue)*1.2])
          self.figs["bievent"].update_yaxes(range=[0, np.max(obs_antinue)*1.2])
        else:
          self.figs["bievent"].update_traces(selector=0, patch=dict(name="Rate Density"))
          self.figs["bievent"].data[0].x = obs_nue[:,0]
          self.figs["bievent"].data[0].y = obs_antinue[:,0]
          self.figs["bievent"].data[1].x = [curr_nue[0],]
          self.figs["bievent"].data[1].y = [curr_antinue[0],]
          
          self.figs["bievent"].update_traces(selector=2, patch=dict(visible=True))
          self.figs["bievent"].update_traces(selector=3, patch=dict(visible=True))
          self.figs["bievent"].data[2].x = obs_nue[:,1]
          self.figs["bievent"].data[2].y = obs_antinue[:,1]
          self.figs["bievent"].data[3].x = [curr_nue[1],]
          self.figs["bievent"].data[3].y = [curr_antinue[1],]
          
          self.figs["bievent"].update_traces(selector=4, patch=dict(visible=True))
          self.figs["bievent"].update_traces(selector=5, patch=dict(visible=True))
          self.figs["bievent"].data[4].x = obs_nue[:,2]
          self.figs["bievent"].data[4].y = obs_antinue[:,2]
          self.figs["bievent"].data[5].x = [curr_nue[2],]
          self.figs["bievent"].data[5].y = [curr_antinue[2],]

          
          numu_rate = beam(Es) * NumuCCTotInterp(Es) * arb_sf
          numub_rate = beam(Es) * NumubCCTotInterp(Es) * arb_sf
          ocd = self.params["delta"]
          self.params["delta"] = 0.7
          nuep, nuebp = Probability_Matter_LBL(Es, self.params["expt_baseline"], self.params, osc_channels=["nue_appearance", "antinue_appearance"])
          self.figs["bievent"].update_xaxes(range=[0, np.max(nuep*numu_rate)*1.75])
          self.figs["bievent"].update_yaxes(range=[0, np.max(numub_rate*nuebp)*1.3])
          self.params["delta"] = ocd

    osc_controls = ["s13sq", "s23sq", "Dmsq32", "Dmsq21", "delta"]
    self.register_callback("bievent", osc_controls, update)
    det_controls = ["expt_baseline"]
    self.register_callback("bievent", det_controls, update)

    def change_beam(v):
      nonlocal beam
      beam = self.gen_beam(beam_selector.value)

      if beam_selector.value == "DUNE":
        self.params["expt_baseline"] = 1300
        self.sliders["expt_baseline"].value = 1300
      elif beam_selector.value == "NOvA":
        self.params["expt_baseline"] = 810
        self.sliders["expt_baseline"].value = 810
      elif beam_selector.value == "T2K":
        self.params["expt_baseline"] = 295
        self.sliders["expt_baseline"].value = 295
        
      update(v)

    beam_controls = ["gaussbeam_mu", "gaussbeam_w"]
    self.register_callback("bievent", beam_controls, change_beam)
    beam_selector.observe(change_beam)
    dointeg_btn.observe(update)
    [energy_sliders[i].observe(update) for i in range(3)]
    return wdgt.HBox([ wdgt.VBox([self.figs["bievent"], self.figs["bievrate"]]), 
                       wdgt.VBox([wdgt.Label("Oscillation Parameters:", style=dict(font_weight="Bold", font_size="14pt")), self.get_control_box(osc_controls),
                                  wdgt.Label("Select Beam Type:", style=dict(font_weight="Bold", font_size="14pt")), beam_selector,
                                  wdgt.Label("Beam Controls:", style=dict(font_weight="Bold", font_size="14pt")), self.get_control_box(beam_controls),
                                  wdgt.Label("Detector Controls:", style=dict(font_weight="Bold", font_size="14pt")), self.get_control_box(det_controls),
                                  wdgt.Label("Energy Selection:", style=dict(font_weight="Bold", font_size="14pt")),
                                  dointeg_btn,
                                  wdgt.VBox([ wdgt.HBox([wdgt.Label(r"${\color{%s}\mathrm{Energy} %s}$" % (self.ecols[i],i)), energy_sliders[i]]) for i in range(3) ], layout=wdgt.Layout(margin_top="10px")) 
                                 ]) 
                     ])

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