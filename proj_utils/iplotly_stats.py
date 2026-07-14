import numpy as np
import plotly.graph_objects as go
import ipywidgets as wdgt
import pandas as pa

from proj_utils.NuFast import *
from proj_utils.stats import *
from proj_utils.xsec import NumuCCTotInterp, NumubCCTotInterp
from proj_utils.flux import DUNENDNumu, NOvANDNumu, SKNumu

from proj_utils.visualize import *

class iPlotlyStats:
  def __init__(self):
    self.figs = {}

    self.ecols = ["#EE7733","#EE3377","#AA4499"]
    self.nucols = ["#AA3377","#4477AA","#EE6677","#66CCEE"]
    self.lines = {
      "numu": dict(color=self.nucols[0]),
      "nue": dict(color=self.nucols[1]),
      "antinumu": dict(color=self.nucols[2]),
      "antinue": dict(color=self.nucols[3]),
      
      "E0": dict(color=self.ecols[0]),
      "E1": dict(color=self.ecols[1]),
      "E2": dict(color=self.ecols[2]),
    }

    self.evnu = pa.read_csv("simulation/neutrino_mode_events.csv")
    self.evnub = pa.read_csv("simulation/antineutrino_mode_events.csv")
    
  def gaus_fit(self):
    true_params = (np.random.default_rng().random(2) * np.array([6,1.5])) + np.array([-3,0])
    curr_params = [0, 1]

    num_samples = max(5,int(np.pow(10, np.random.default_rng().random(1) * 5)))
    bins = np.linspace(start=-3, stop=3, num=200)
    bin_centers = (bins[1:] + bins[:-1])/2.0
    
    gaus_x, gaus_y = get_scaled_gaus(num_samples, loc=curr_params[0], scale=curr_params[1], bins=bins)

    def _get_hist(gaus_params, num_samples, bins):
      return hist1dtoline(*hist1d(data=np.random.default_rng().normal(loc=gaus_params[0], 
                                                                      scale=gaus_params[1], 
                                                                      size=num_samples), 
                                  bins=bins)
                         )

    h_obs = _get_hist(true_params, num_samples, bins=bins)
    
    self.figs["gaus_fit"] = go.FigureWidget(
       data=[ 
               go.Scatter(x=gaus_x, 
                          y=gaus_y,
                          name="g(x | mu, sigma)"),
               go.Scatter(x=h_obs[0],
                          y=h_obs[1],
                          name=f'Observation N={num_samples}',
                          marker_color='rgb(55, 83, 109)'
                )
            ],
       layout=go.Layout(template="simple_white", height=480, width=480,
                        yaxis_title='Count',
                        xaxis_title='x',
                        margin=dict(b=10,l=10,t=5,r=5),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ))
    )
    self.figs["gaus_fit"].update_yaxes(range=[0, np.max(h_obs)*1.2])
    self.figs["gaus_fit"].update_xaxes(range=[-3,3])

    mu_slider = wdgt.FloatSlider(orientation='horizontal',
                                 value=curr_params[0],
                                 min=-3,
                                 max=3,
                                 step=0.025)

    sigma_slider = wdgt.FloatSlider(orientation='horizontal',
                                    value=curr_params[1],
                                    min=0,
                                    max=1.5,
                                    step=0.025)

    sample_btn = wdgt.Button(
        description='Draw New Sample',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        icon='dice-d20' # (FontAwesome names without the `fa-` prefix)
    )
    reveal_button = wdgt.Button(
        description='Reveal Parameters',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        icon='display' # (FontAwesome names without the `fa-` prefix)
    )

    def update(v):
      curr_params = [mu_slider.value, sigma_slider.value]

      with self.figs["gaus_fit"].batch_update():
        self.figs["gaus_fit"].data[0].y = get_scaled_gaus(num_samples, loc=curr_params[0], scale=curr_params[1], bins=bins)[1]

    def resample(v):
      with self.figs["gaus_fit"].batch_update():
        nonlocal true_params, num_samples
        true_params = (np.random.default_rng().random(2) * np.array([6,1.5])) + np.array([-3,0])
        num_samples = max(5, int(np.pow(10, np.random.default_rng().random(1) * 5)))
        h_obs = _get_hist(true_params, num_samples, bins=bins)
        self.figs["gaus_fit"].data[1].y = h_obs[1]
        self.figs["gaus_fit"].update_traces(selector=1, 
                                            patch=dict(name=f"Observation N={num_samples}"))  
        self.figs["gaus_fit"].update_yaxes(range=[0, np.max(h_obs)*1.2])

      update(v)

    def reveal(v):
      with self.figs["gaus_fit"].batch_update():
        self.figs["gaus_fit"].update_traces(selector=1, 
                patch=dict(name=f"Observation N={num_samples}, True mu={true_params[0]:.2g}, sigma={true_params[1]:.2g})"))
    
    mu_slider.observe(update)
    sigma_slider.observe(update)
    sample_btn.on_click(resample)
    reveal_button.on_click(reveal)
    
    return wdgt.HBox([self.figs["gaus_fit"],
                      wdgt.VBox([ 
                        wdgt.HBox([wdgt.Label("Gaussian Location:", layout=wdgt.Layout(width="10em")), mu_slider ]),
                        wdgt.HBox([wdgt.Label("Gaussian Scale:", layout=wdgt.Layout(width="10em")), sigma_slider ]),
                        wdgt.HBox([sample_btn, reveal_button])
                      ])
                     ])

  def get_osc_hist(self, evdf, osc_params, bins, osc_channels):
    bv, be = hist1d(data=evdf["E_neutrino"], 
                                weights=Probability_Matter_LBL(evdf["E_neutrino"], 1300, 
                                                               osc_params, osc_channels=osc_channels), 
                                bins=bins)
    return hist1dtoline(bv, be), ((be[1:] + be[:-1])/2.0, bv)

  def hist_comp1(self): 
    osc_params = {
      "s12sq": 0.31,
      "s13sq": 0.02,
      "s23sq": 0.55,
      "delta": 0.7 * np.pi,
      "Dmsq21": 7.5e-5,
      "Dmsq31": 2.5e-3
    }
  
    events_nu_nue_cc = self.evnu[self.evnu["pid_lepton"] == 11]

    start = 0
    stop = 8
    nbins = 15
    num = nbins + 1
            
    h1,_ = self.get_osc_hist(events_nu_nue_cc, osc_params, np.linspace(start, stop, num),["nue_appearance"])
    osc_params["Dmsq31"] = 2.8e-3
    h2,_ = self.get_osc_hist(events_nu_nue_cc, osc_params, np.linspace(start, stop, num),["nue_appearance"])

    return go.FigureWidget(
       data=[ 
               go.Scatter(x=h1[0], 
                          y=h1[1],
                          line_color=self.nucols[0],
                          name="Hist A"),
               go.Scatter(x=h2[0],
                          y=h2[1],
                          line_color=self.nucols[1],
                          name="Hist B")
            ],
       layout=go.Layout(template="simple_white", height=480, width=480,
                        yaxis_title='Count',
                        xaxis_title='x',
                        margin=dict(b=10,l=10,t=5,r=5),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ))
          )

  def hist_comp2(self):
    osc_params = {
      "s12sq": 0.31,
      "s13sq": 0.02,
      "s23sq": 0.55,
      "delta": 0.7 * np.pi,
      "Dmsq21": 7.5e-5,
      "Dmsq31": 2.5e-3
    }
  
    events_nu_nue_cc = self.evnu[self.evnu["pid_lepton"] == 11]

    start = 0
    stop = 8
    nbins = 15
    num = nbins + 1
            
    h1,_ = self.get_osc_hist(events_nu_nue_cc, osc_params, np.linspace(start, stop, num),["nue_appearance"])
    osc_params["s13sq"] = 0.021
    h3,_ = self.get_osc_hist(events_nu_nue_cc, osc_params, np.linspace(start, stop, num),["nue_appearance"])

    return go.FigureWidget(
       data=[ 
               go.Scatter(x=h1[0], 
                          y=h1[1],
                          line_color=self.nucols[0],
                          name="Hist A"),
               go.Scatter(x=h3[0],
                          y=h3[1],
                          line_color=self.nucols[2],
                          name="Hist C")
            ],
       layout=go.Layout(template="simple_white", height=480, width=480,
                        yaxis_title='Count',
                        xaxis_title='x',
                        margin=dict(b=10,l=10,t=5,r=5),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ))
          )

    
  def hist_comp3(self):
    osc_params = {
      "s12sq": 0.31,
      "s13sq": 0.02,
      "s23sq": 0.55,
      "delta": 0.7 * np.pi,
      "Dmsq21": 7.5e-5,
      "Dmsq31": 2.5e-3
    }
  
    events_nu_nue_cc = self.evnu[self.evnu["pid_lepton"] == 11]

    start = 0
    stop = 8
    nbins = 15
    num = nbins + 1
            
    h1, _ = self.get_osc_hist(events_nu_nue_cc, osc_params, np.linspace(start, stop, num),["nue_appearance"])
    osc_params["Dmsq31"] = 2.8e-3
    h2, h2r = self.get_osc_hist(events_nu_nue_cc, osc_params, np.linspace(start, stop, num),["nue_appearance"])
    osc_params["Dmsq31"] = 2.5e-3
    osc_params["s13sq"] = 0.021
    h3, h3r = self.get_osc_hist(events_nu_nue_cc, osc_params, np.linspace(start, stop, num),["nue_appearance"])

    return wdgt.HBox([ go.FigureWidget(
         data=[ 
                 go.Scatter(x=h1[0], 
                            y=h1[1],
                            line_color=self.nucols[0],
                            name="Hist A"),
                 go.Scatter(x=h2[0],
                            y=h2[1],
                            line_color=self.nucols[1],
                            name="Hist B"),
                 go.Scatter(x=h2r[0],
                            y=h2r[1],
                            mode="markers",
                            line_color=self.nucols[1],
                            marker_color=self.nucols[1],
                            error_y=dict(
                              type='data', # value of error bar given in data coordinates
                              array=np.sqrt(h2r[1]/4),
                              visible=True),
                            showlegend=False)
              ],
         layout=go.Layout(template="simple_white", height=480, width=480,
                          yaxis_title='Count',
                          xaxis_title='x',
                          margin=dict(b=10,l=10,t=5,r=5),
                          legend=dict(
                              orientation="h",
                              yanchor="top",
                              y=0.99,
                              xanchor="right",
                              x=0.99
                          ))
            ),
        go.FigureWidget(
         data=[ 
                 go.Scatter(x=h1[0], 
                            y=h1[1],
                            line_color=self.nucols[0],
                            name="Hist A"),
                 go.Scatter(x=h3[0],
                            y=h3[1],
                            line_color=self.nucols[2],
                            name="Hist C"),
                 go.Scatter(x=h3r[0],
                            y=h3r[1],
                            line_color=self.nucols[2],
                            marker_color=self.nucols[2],
                            mode="markers",
                            error_y=dict(
                              type='data', # value of error bar given in data coordinates
                              array=np.sqrt(h3r[1]*2),
                              visible=True),
                            showlegend=False)
              ],
         layout=go.Layout(template="simple_white", height=480, width=480,
                          yaxis_title='Count',
                          xaxis_title='x',
                          margin=dict(b=10,l=10,t=5,r=5),
                          legend=dict(
                              orientation="h",
                              yanchor="top",
                              y=0.99,
                              xanchor="right",
                              x=0.99
                          ))
            ),
           ])

  def lhood_scan_1(self):

    osc_params = {
      "s12sq": 0.31,
      "s13sq": 0.02,
      "s23sq": 0.55,
      "delta": 0.7 * np.pi,
      "Dmsq21": 7.5e-5,
      "Dmsq31": 2.64e-3
    }
  
    events_nu_numu_cc = self.evnu[self.evnu["pid_lepton"] == 13]

    bins = np.linspace(start=0, stop=8, num=50)
    
    hobs, _ = self.get_osc_hist(events_nu_numu_cc, osc_params, bins, ["numu_survival"])

    osc_params["Dmsq31"] = 2.5e-3
    hpred, herr = self.get_osc_hist(events_nu_numu_cc, osc_params, bins, ["numu_survival"])
    
    self.figs["lhood_scan_1"] = go.FigureWidget(
       data=[ 
               go.Scatter(x=hpred[0],
                          y=hpred[1], 
                          line_color=self.nucols[1],
                          name="Prediction"),
               go.Scatter(x=herr[0],
                          y=herr[1],
                          line_color=self.nucols[1],
                          marker_color=self.nucols[1],
                          mode="markers",
                          error_y=dict(
                            type='data', # value of error bar given in data coordinates
                            array=np.sqrt(herr[1]),
                            visible=True),
                          showlegend=False),
         
               go.Scatter(x=hobs[0], 
                          y=hobs[1],
                          line_color=self.nucols[0],
                          name="Observation"),
            ],
       layout=go.Layout(template="simple_white", height=480, width=480,
                        yaxis_title='Count',
                        xaxis_title='x',
                        margin=dict(b=10,l=10,t=5,r=5),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ))
    )

    dm31_slider = wdgt.FloatSlider(orientation='horizontal',
                                  value=osc_params["Dmsq31"]*1E3,
                                  min=2,
                                  max=3,
                                  step=0.025)

    def update(v):
      osc_params["Dmsq31"] = dm31_slider.value*1E-3
      hpred, herr = self.get_osc_hist(events_nu_numu_cc, osc_params, bins, ["numu_survival"])
    
      with self.figs["lhood_scan_1"].batch_update():
        self.figs["lhood_scan_1"].data[0].y = hpred[1]
        self.figs["lhood_scan_1"].data[1].y = herr[1]
        self.figs["lhood_scan_1"].update_traces(selector=1, patch=dict(error_y_array=np.sqrt(herr[1])))

    dm31_slider.observe(update)
    
    return wdgt.HBox([self.figs["lhood_scan_1"],
                      wdgt.VBox([ 
                        wdgt.HBox([wdgt.Label(r"$\Delta{}m^{2}_{31} [10^{-3} eV^{2}]$", layout=wdgt.Layout(width="10em")), dm31_slider ]),
                      ])
                     ])
