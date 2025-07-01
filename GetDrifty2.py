import ANNarchy as ann
import math
import matplotlib.pyplot as plt
import numpy as np
from plotting_widget import *
from Utilities import get_active_neurons, generate_random_pattern
LIneuron = ann.Neuron(
   parameters = dict(
        tau = 50.0,
        baseline = 0,
        I_ext = ann.Parameter(0.0),
        # ex =ann.Parameter(0.0),
        ex_0 = ann.Parameter(0.0),
        tau_e = 12*60,
        E = 3.5,
        r_th = 0.4
    ),
    equations = [
        'tau_e * dex/dt + ex - ex_0  = if r > r_th: (ex_0/absolute(ex_0)) * E  else: 0',
        'tau * dr/dt + r =  ex*tanh(sum(target) + I_ext )'
    ],
    functions = """
        absolute(x) = (x*x)**0.5,
    """
) 

E_synapse = ann.Synapse(
    parameters = dict(
        tau = 1,
        lr_e = 0.45,
        dr_e = 0.55,
        min_w = -1,
        max_w = 1,
        act_thrsh = 0.4
    ),
    equations = [
        'pr = absolute(pre.r)',
        'po_r = absolute(post.r)',
        'tau*dw/dt + dr_e * w =  pre_post_lr(pre.r,post.r) * lr_e * (1-w) * pre.r * post.r: min = min_w, max = max_w',
    ],
    functions = """
        absolute(x) = (x*x)**0.5,
        pre_post_lr(pr,po_r) = if ((pr > 0) and (po_r > 0)): 1 else: 0
    """
)


num_HPC_E_neuron = 70

net = ann.Network()
E_pop = net.create(geometry=num_HPC_E_neuron, neuron=LIneuron, name='E_pop')
EE_proj = net.connect(E_pop,E_pop,'exc',E_synapse)
EE_proj.connect_all_to_all(weights=ann.Uniform(0.0,0.01),allow_self_connections=True)

# output_neuron = net.create(geometry=1, neuron=OPneuron, name='op_pop')
# Eop_proj = net.connect(E_pop,output_neuron,'exc',HomeoE_synapse)
# Eop_proj.connect_all_to_all(weights=ann.Uniform(0.0,0.02))

net.compile(clean=True)


# Simulation parameters
Nrep = 1 # number of repetitions
T = 500 # duration of repetitions
# IR = 100 # Inter-repetition interval
del_t = 0.1 # time step


num_days = 1 # number of days in the simulation
# ID = 1000 #inter-day delay
# delta = 20 # input current
theta = 5 #threshold firing rate for active neurons
c = 1 #cap pn recurrent weights
E = 0.6 # Epsi_i increase factor for neurons
# normal distribution parameters for Epsi_i
mu = 0
sigma = 1
net.config(dt=del_t)

PAT_LEN = 10
np.random.seed(2025)
PATTERN_A = np.concat((generate_random_pattern(PAT_LEN),np.zeros(num_HPC_E_neuron-PAT_LEN)))
print(PATTERN_A)
baseline_epsi = np.random.uniform(mu, sigma, num_HPC_E_neuron)
E_pop[:].I_ext = PATTERN_A
E_pop[:].ex_0 = baseline_epsi
act_monitor = net.monitor(E_pop,'r')
ex_monitor = net.monitor(E_pop,'ex')
rec_weights = []
rec_weights.append(EE_proj.connectivity_matrix())
net.simulate(T)
net_activity = act_monitor.get('r').T
excitability = ex_monitor.get('ex').T
rec_weights.append(EE_proj.connectivity_matrix())
plot_weights_over_time(rec_weights,[ "Before" ,"After"] ,"./plots/drifty2/EE_weights.png")

plot_activity_n_excitability_time([net_activity,excitability],
                       titles=['Neuronal Activity',
                               'Excitability',],
                       fname="./plots/drifty2/NetActivity.png",
                       cmaps=['Blues', 'Greens', 'Reds'])


# test phase

breakpoint()