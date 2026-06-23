import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

def _pois_interval_Wald(c):
    return c - newton(lambda ll: ll + np.sqrt(ll) - c,c), newton(lambda lh: lh - np.sqrt(lh) - c,c) - c

def _pois_intervals(counts):
    l = []
    h = []
    for c in counts:
        a, b = pois_interval_Wald(c)
        l.append(a)
        h.append(b)
    return [l,h]

def hist1d(data, bins, weights=None):
    if weights is None:
        bin_vals, bins = np.histogram(data, bins=bins) 
    else:
        bin_vals, bins = np.histogram(data, bins=bins, weights=weights)
    return (bin_vals, bins)

def write_hist1d(hist, filename):
  np.save(filename, np.array([bin_vals, bins]))

def read_hist1d(filename):
  npd = numpy.load(filename)
  return npd[0], ndp[1]

def poisson_fluctuate(counts):
    rng = np.random.default_rng()
    return rng.poisson(counts)

def drawhist1d(*, hist=None, data=None, bins=None, weights=None, **kwargs):
    if data is not None:
        counts, bins = hist1d(data, bins, weights)
    elif hist is not None:
        counts = hist[0]
        bins = hist[1]
    else:
        raise RuntimeError("Must pass either hist= or data= arguments")

    plt.gca().stairs(counts, bins, **kwargs)

def ReconstructedNeutrinoEnergy(df, params={}):

  muon_Eres = params["muon_energy_resolution"] if "muon_energy_resolution" in params else 0
  electron_Eres = params["electron_energy_resolution"] if "electron_energy_resolution" in params else 0
  proton_Tres = params["proton_kinetic_energy_resolution"] if "proton_kinetic_energy_resolution" in params else 0
  cpi_Tres = params["charged_pion_kinetic_energy_resolution"] if "charged_pion_kinetic_energy_resolution" in params else 0
  cpi_Mfrac = params["charged_pion_mass_visible_fraction"] if "charged_pion_mass_visible_fraction" in params else 1
  pi0_Eres = params["neutral_pion_energy_resolution"] if "neutral_pion_energy_resolution" in params else 0
  
  erec = 0
  if muon_Eres > 0:
      erec += np.where(df["pid_lep"] == 13, np.random.normal(df["E_lepton"], muon_Eres*df["E_lepton"]), 0)
  else:
      erec += np.where(df["pid_lep"] == 13, df["E_lepton"], 0)

  if electron_Eres > 0:
      erec += np.where(df["pid_lep"] == 11, np.random.normal(df["E_lepton"], electron_Eres*df["E_lepton"]),0)
  else:
      erec += np.where(df["pid_lep"] == 11, df["E_lepton"], 0)
  
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

def Poisson_N2LLH(data_hist, predicted_hist):
    pred_no0 = predicted_hist[0][data_hist[0] != 0]
    data_no0 = data_hist[0][data_hist[0] != 0]
    return 2*np.sum(pred_no0 - data_no0 + data_no0 * np.log(data_no0/pred_no0))

##### NuFast below

# Some constants
eVsqkm_to_GeV_over4 = 1e-9 / 1.97327e-7 * 1e3 / 4
YerhoE2a = 1.52588e-4
Yerho = 0.5 * 3.0
N_Newton = 0

# Probability_Matter_LBL calculates all nine oscillation probabilities including
# the matter effect in an optimized, fast, and efficient way. The precision can
# be controlled with N_Newton. For many applications N_Newton=0 may be enough,
# but many years of DUNE or HK-LBL may require N_Newton=1. This code may be
# suitable for atmospheric neutrinos. The code is standalone with the Parameters
# module for several constants.
#
# Inputs:
#   mixing angles (usual parameterization)
#   phase (usual parameterization) make Dmsq31 positive/negative for the NO/IO
#   Delta msq's (eV^2)
#   L (km)
#   E (GeV) positive for neutrinos, negative for antineutrinos
#   rho (g/cc)
#   Ye: electron fraction, typically around 0.5
#   N_Newton: number of Newton's method iterations to do. should be zero, one, two (or higher)
# Outputs:
#   probs_returned is all nine oscillation probabilities: e.g. probs_returned[1][0] is mu->e
def Probability_Matter_LBL(E, osc_params, osc_channels=[]):
  # --------------------------------------------------------------------- #
  # First calculate useful simple functions of the oscillation parameters #
  # --------------------------------------------------------------------- #

  L = osc_params["experimental_baseline_km"]
  s12sq = osc_params["s12sq"]
  s13sq = osc_params["s13sq"]
  s23sq = osc_params["s23sq"]
  delta = osc_params["delta"]
  Dmsq21 = osc_params["Dmsq21"]
  Dmsq31 = osc_params["Dmsq31"]

  osc_channels_id = []

  for oc in osc_channels:
    if oc == "numu_survival":
      osc_channels_id.append((1,1))
    if oc == "nue_appearance":
      osc_channels_id.append((1,0))
  
  c13sq = 1 - s13sq

  # Ueisq's
  Ue2sq = c13sq * s12sq
  Ue3sq = s13sq

  # Umisq's, Utisq's and Jvac  
  Um3sq = c13sq * s23sq
  # Um2sq and Ut2sq are used here as temporary variables, will be properly defined later   
  Ut2sq = s13sq * s12sq * s23sq
  Um2sq = (1 - s12sq) * (1 - s23sq)

  Jrr = np.sqrt(Um2sq * Ut2sq)
  sind = np.sin(delta)
  cosd = np.cos(delta)

  Um2sq = Um2sq + Ut2sq - 2 * Jrr * cosd
  Jmatter = 8 * Jrr * c13sq * sind
  Amatter = Yerho * E * YerhoE2a
  Dmsqee = Dmsq31 - s12sq * Dmsq21

  # calculate A, B, C, See, Tee, and part of Tmm
  A = Dmsq21 + Dmsq31 # temporary variable
  See = A - Dmsq21 * Ue2sq - Dmsq31 * Ue3sq
  Tmm = Dmsq21 * Dmsq31 # using Tmm as a temporary variable   
  Tee = Tmm * (1 - Ue3sq - Ue2sq)
  C = Amatter * Tee
  A = A + Amatter

  # ---------------------------------- #
  # Get lambda3 from lambda+ of MP/DMP #
  # ---------------------------------- #
  xmat = Amatter / Dmsqee
  tmp = 1 - xmat
  lambda3 = Dmsq31 + 0.5 * Dmsqee * (xmat - 1 + np.sqrt(tmp * tmp + 4 * s13sq * xmat))

  # ---------------------------------------------------------------------------- #
  # Newton iterations to improve lambda3 arbitrarily, if needed, (B needed here) #
  # ---------------------------------------------------------------------------- #
  B = Tmm + Amatter * See # B is only needed for N_Newton >= 1
  for i in range(N_Newton):
    lambda3 = (lambda3 * lambda3 * (lambda3 + lambda3 - A) + C) / (lambda3 * (2 * (lambda3 - A) + lambda3) + B) # this strange form prefers additions to multiplications

  # ------------------- #
  # Get  Delta lambda's #
  # ------------------- #
  tmp = A - lambda3
  Dlambda21 = np.sqrt(tmp * tmp - 4 * C / lambda3)
  lambda2 = 0.5 * (A - lambda3 + Dlambda21)
  Dlambda32 = lambda3 - lambda2
  Dlambda31 = Dlambda32 + Dlambda21

  # ----------------------- #
  # Use Rosetta for Veisq's #
  # ----------------------- #
  # denominators    
  PiDlambdaInv = 1 / (Dlambda31 * Dlambda32 * Dlambda21)
  Xp3 = PiDlambdaInv * Dlambda21
  Xp2 = -PiDlambdaInv * Dlambda31

  # numerators
  Ue3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3
  Ue2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2

  Smm = A - Dmsq21 * Um2sq - Dmsq31 * Um3sq
  Tmm = Tmm * (1 - Um3sq - Um2sq) + Amatter * (See + Smm - A)

  Um3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3
  Um2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2

  # ------------- #
  # Use NHS for J #
  # ------------- #
  Jmatter = Jmatter * Dmsq21 * Dmsq31 * (Dmsq31 - Dmsq21) * PiDlambdaInv

  # ----------------------- #
  # Get all elements of Usq #
  # ----------------------- #
  Ue1sq = 1 - Ue3sq - Ue2sq
  Um1sq = 1 - Um3sq - Um2sq

  Ut3sq = 1 - Um3sq - Ue3sq
  Ut2sq = 1 - Um2sq - Ue2sq
  Ut1sq = 1 - Um1sq - Ue1sq

  # ----------------------- #
  # Get the kinematic terms #
  # ----------------------- #
  Lover4E = eVsqkm_to_GeV_over4 * L / E

  D21 = Dlambda21 * Lover4E
  D32 = Dlambda32 * Lover4E
    
  sinD21 = np.sin(D21)
  sinD31 = np.sin(D32 + D21)
  sinD32 = np.sin(D32)

  triple_sin = sinD21 * sinD31 * sinD32

  sinsqD21_2 = 2 * sinD21 * sinD21
  sinsqD31_2 = 2 * sinD31 * sinD31
  sinsqD32_2 = 2 * sinD32 * sinD32

  # ------------------------------------------------------------------- #
  # Calculate the three necessary probabilities, separating CPC and CPV #
  # ------------------------------------------------------------------- #
  Pme_CPC = (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2 \
        + (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2 \
        + (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * sinsqD32_2
  Pme_CPV = -Jmatter * triple_sin

  Pmm = 1 - 2 * (Um2sq * Um1sq * sinsqD21_2 \
         + Um3sq * Um1sq * sinsqD31_2 \
         + Um3sq * Um2sq * sinsqD32_2)

  Pee = 1 - 2 * (Ue2sq * Ue1sq * sinsqD21_2 \
         + Ue3sq * Ue1sq * sinsqD31_2 \
         + Ue3sq * Ue2sq * sinsqD32_2)
    
  probs_returned = []

  for ch in osc_channels_id:
      if ch[0] == 0:
          if ch[1] == 0:
              probs_returned.append(Pee)
          elif ch[1] == 1:
              probs_returned.append(Pme_CPC - Pme_CPV)
          elif ch[1] == 2:
              probs_returned.append(1 - Pee - (Pme_CPC - Pme_CPV))
      if ch[0] == 1:
          if ch[1] == 0:
              probs_returned.append(Pme_CPC + Pme_CPV)
          elif ch[1] == 1:
              probs_returned.append(Pmm)
          elif ch[1] == 2:
              probs_returned.append(1 - (Pme_CPC + Pme_CPV) - Pmm)
      if ch[0] == 2:
          if ch[1] == 0:
              probs_returned.append(1 - Pee - (Pme_CPC + Pme_CPV))
          elif ch[1] == 1:
              probs_returned.append(1 - (Pme_CPC - Pme_CPV) - Pmm)
          elif ch[1] == 2:
              probs_returned.append(1 - (1 - Pee - (Pme_CPC - Pme_CPV)) - (1 - (Pme_CPC + Pme_CPV) - Pmm))
  return probs_returned