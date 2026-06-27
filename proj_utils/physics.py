import numpy as np

def ReconstructedNeutrinoEnergy(df, params={}):

  muon_Eres = params["muon_energy_resolution"] if "muon_energy_resolution" in params else 0
  electron_Eres = params["electron_energy_resolution"] if "electron_energy_resolution" in params else 0
  proton_Tres = params["proton_kinetic_energy_resolution"] if "proton_kinetic_energy_resolution" in params else 0
  cpi_Tres = params["charged_pion_kinetic_energy_resolution"] if "charged_pion_kinetic_energy_resolution" in params else 0
  cpi_Mfrac = params["charged_pion_mass_visible_fraction"] if "charged_pion_mass_visible_fraction" in params else 1
  pi0_Eres = params["neutral_pion_energy_resolution"] if "neutral_pion_energy_resolution" in params else 0
  
  erec = 0
  if muon_Eres > 0:
      erec += np.where(df["pid_lepton"] == 13, np.random.normal(df["E_lepton"], muon_Eres*df["E_lepton"]), 0)
  else:
      erec += np.where(df["pid_lepton"] == 13, df["E_lepton"], 0)

  if electron_Eres > 0:
      erec += np.where(df["pid_lepton"] == 11, np.random.normal(df["E_lepton"], electron_Eres*df["E_lepton"]),0)
  else:
      erec += np.where(df["pid_lepton"] == 11, df["E_lepton"], 0)
  
  if proton_Tres > 0:
      erec += np.random.normal(df["T_proton"], proton_Tres*df["T_proton"])
  else:
      erec += df["T_proton"]
  
  if pi0_Eres > 0:
      erec += np.random.normal(df["E_pi0"], pi0_Eres*df["E_pi0"])
  else:
      erec += df["E_pi0"]
  
  if cpi_Tres > 0:
      erec += np.random.normal(df["T_cpi"], cpi_Tres*df["T_cpi"])
  else:
      erec += df["T_cpi"]
  
  erec += df["num_cpi"] * 0.1395 * cpi_Mfrac
  return erec
