import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import Button, FloatSlider, Label, VBox, HBox, Layout, Output
import pandas as pa

from scipy.stats import norm

from .stats import *
from .visualize import *

def _get_hist(gaus_params, num_samples, bins):
  return hist1d(data=np.random.default_rng().normal(loc=gaus_params[0], scale=gaus_params[1], size=num_samples), bins=bins)[0]

InteractiveGausFitfig = None
def InteractiveGausFit():
  global InteractiveGausFitfig
  if InteractiveGausFitfig is not None:
    plt.close(InteractiveGausFitfig)

  true_params = (np.random.default_rng().random(2) * np.array([6,1.5])) + np.array([-1.5,0])
  num_samples = int(np.pow(10, np.random.default_rng().random(1) * 5))
  bins = np.linspace(start=-3, stop=3, num=200)
  
  plt.ioff()
  InteractiveGausFitfig, ax = plt.subplots(figsize=(8,4))
  InteractiveGausFitfig.canvas.toolbar_visible = False

  curr_params = [0, 1]
  gaus_x, gaus_y = get_scaled_gaus(num_samples, loc=curr_params[0], scale=curr_params[1], bins=bins)
  gaus_l, = ax.plot(gaus_x, gaus_y, label="g(x | location, scale)")

  h_obs = _get_hist(true_params, num_samples, bins=bins)
  obs_l = ax.stairs(h_obs, bins, lw=2, label=f"Observation N={num_samples}", color="#AA3377")

  ax.set_xlabel(r"$x$", size="large")
  ax.set_ylabel(r"Count", size="x-large")
  ax.set_ylim([0, np.max(h_obs)*1.25])
  ax.legend()

  controls_spec = [
    dict( pname = "loc", v=curr_params[0], srange = [-3,3], sf = 1, desc = r"Location" ),
    dict( pname = "scl", v=curr_params[1], srange = [0.01,2], sf = 1, desc = r"Scale" ),
  ]
  
  sliders = []
  controls = []

  for c in controls_spec:
    sliders.append(FloatSlider(orientation='horizontal',
                               value=c["v"],
                               min=c["srange"][0],
                               max=c["srange"][1],
                               step=(c["srange"][1]-c["srange"][0])/200
                              ))
    controls.append(HBox([Label(c["desc"]), sliders[-1]]))

  sample_btn = Button(
      description='Draw New Sample',
      disabled=False,
      button_style='info', # 'success', 'info', 'warning', 'danger' or ''
      icon='dice-d20' # (FontAwesome names without the `fa-` prefix)
  )
  controls.append(sample_btn)

  reveal_button = Button(
      description='Reveal Parameters',
      disabled=False,
      button_style='info', # 'success', 'info', 'warning', 'danger' or ''
      icon='display' # (FontAwesome names without the `fa-` prefix)
  )
  controls.append(reveal_button)
  
  updating = False
  def update(val):
    nonlocal updating, curr_params
    if updating:
      return
    updating = True

    curr_params[0] = sliders[0].value
    curr_params[1] = sliders[1].value
      
    gaus_l.set_ydata(get_scaled_gaus(num_samples, loc=curr_params[0], scale=curr_params[1], bins=bins)[1])
    
    InteractiveGausFitfig.canvas.draw_idle()

    updating = False

  def draw_sample(val):
    nonlocal updating, true_params, num_samples, obs_l, ax
    if updating:
      return
    updating = True

    true_params = (np.random.default_rng().random(2) * np.array([6,1.5])) + np.array([-1.5,0])
    num_samples = int(np.pow(10, 0.1 + np.random.default_rng().random(1) * 5))
      
    obs_l.remove()
    h_obs = _get_hist(true_params, num_samples, bins=bins)
    obs_l = ax.stairs(h_obs, bins, lw=2, label=f"Observation N={num_samples}", color="#AA3377")
    gaus_l.set_ydata(get_scaled_gaus(num_samples, loc=curr_params[0], scale=curr_params[1], bins=bins)[1])
    ax.set_ylim([0, np.max(h_obs)*1.25])
    ax.legend()

    InteractiveGausFitfig.canvas.draw_idle()

    updating = False

  def reveal(val):
    nonlocal obs_l, ax
    obs_l.set_label(f"Observation N={num_samples}, True = (loc={true_params[0]:.2g}, scl={true_params[1]:.2g})")
    ax.legend()

    InteractiveGausFitfig.canvas.draw_idle()
  
  for s in sliders:
    s.observe(update)
  
  display(InteractiveGausFitfig.canvas)
  for c in controls:
    display(c)

  sample_btn.on_click(draw_sample)
  reveal_button.on_click(reveal)