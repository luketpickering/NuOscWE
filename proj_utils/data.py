import numpy as np
import pandas as pa

from proj_utils.NuFast import *
from proj_utils.stats import *
from proj_utils.physics import *
from proj_utils.visualize import *

class AnalysisData:
  def __init__(self):
    self.bins = np.linspace(start=0, stop=8, num=40)
    self.fns = [ "data/obs_nu_mode_numucc.csv",
              "data/obs_nu_mode_nuecc",
              "data/obs_antinu_mode_numucc",
              "data/obs_antinu_mode_nuecc" ]

  def get_bins(self):
    return self.bins

  def get_obs_nu_mode_numucc(self):
    return read_hist1d(self.fns[0])
  def get_obs_nu_mode_nuecc(self):
    return read_hist1d(self.fns[1])
  def get_obs_antinu_mode_numucc(self):
    return read_hist1d(self.fns[2])
  def get_obs_antinu_mode_nuecc(self):
    return read_hist1d(self.fns[3])
  
  def gen(self):
    sf = 0.15
    
    osc_params = {
      "s12sq": 0.31,
      "s13sq": 0.018,
      "s23sq": 0.535,
      "delta": 0.15 * np.pi,
      "Dmsq21": 7.5e-5,
      "Dmsq31": 2.435e-3
    }
    
    reco_params = {  "muon_energy_resolution": 0.02, # 2%
                     "electron_energy_resolution": 0.05, #
                     "proton_kinetic_energy_resolution": 0.1, #
                     "charged_pion_kinetic_energy_resolution": 0.1, #
                     "charged_pion_mass_visible_fraction": 0.75, #
                     "neutral_pion_energy_resolution": 0.1 #
                  }


    evnu = pa.read_csv("simulation/neutrino_mode_events.csv")
    evnub = pa.read_csv("simulation/antineutrino_mode_events.csv")

    DUNE_baseline = 1300 # km

    events_nu_numu_cc = evnu[evnu["pid_lepton"] == 13]
    events_nu_nue_cc = evnu[evnu["pid_lepton"] == 11]
    events_nu_antinumu_cc = evnub[evnub["pid_lepton"] == -13]
    events_nu_antinue_cc = evnub[evnub["pid_lepton"] == -11]
    
    reco_E_nu_numu = ReconstructedNeutrinoEnergy(events_nu_numu_cc, reco_params)
    reco_E_nu_nue = ReconstructedNeutrinoEnergy(events_nu_nue_cc, reco_params)
    reco_E_antinu_numu = ReconstructedNeutrinoEnergy(events_nu_antinumu_cc, reco_params)
    reco_E_antinu_nue = ReconstructedNeutrinoEnergy(events_nu_antinue_cc, reco_params)

    event_osc_weights_numu_surv = \
        Probability_Matter_LBL(events_nu_numu_cc["E_neutrino"], DUNE_baseline, 
                               osc_params, osc_channels=["numu_survival"])
    event_osc_weights_nue_app = \
        Probability_Matter_LBL(events_nu_nue_cc["E_neutrino"], DUNE_baseline, 
                               osc_params, osc_channels=["nue_appearance"])
    
    event_osc_weights_antinumu_surv = \
        Probability_Matter_LBL(events_nu_antinumu_cc["E_neutrino"], DUNE_baseline, 
                               osc_params, osc_channels=["antinumu_survival"])
    event_osc_weights_antinue_app = \
        Probability_Matter_LBL(events_nu_antinue_cc["E_neutrino"], DUNE_baseline, 
                               osc_params, osc_channels=["antinue_appearance"])
    
    hists = [ hist1d(data=reco_E_nu_numu, weights=event_osc_weights_numu_surv, bins=self.bins), \
              hist1d(data=reco_E_nu_nue, weights=event_osc_weights_nue_app, bins=self.bins), \
              hist1d(data=reco_E_antinu_numu, weights=event_osc_weights_antinumu_surv, bins=self.bins), \
              hist1d(data=reco_E_antinu_nue, weights=event_osc_weights_antinue_app, bins=self.bins) ]

    for i,h in enumerate(hists):
      write_hist1d((poisson_fluctuate(h[0]*sf),h[1]), self.fns[i])

      drawhist1d(hist=h)
