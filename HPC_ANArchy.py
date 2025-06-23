import ANNarchy as ann
import matplotlib.pyplot as plt
import numpy as np
# import os
from plotting_widget import before_after_weights, plot_activity,plot_activity_n_excitability_time
from Utilities import generate_random_pattern
HPC_LIneuron = ann.Neuron(
    parameters = dict(
        tau = 50.0,
        baseline = 0,
        I_ext = ann.Parameter(0.1),
        excitability = 0.0,
    ),
    equations = [
        'tau * dr/dt  + r = baseline + tanh(sum(target) + I_ext + excitability)'
    ]
) 

E_synapse = ann.Synapse(
    parameters = dict(
        tau = 1,
        lr_e = 0.004,
        dr_e = ann.Parameter(0),
        min_w = 0.0,
        max_w = 10.0
    ),
    equations = [
        'tau * dw/dt = -w * dr_e + lr_e*pre.r * post.r : min = min_w, max = max_w',
    ]
)


I_synapse = ann.Synapse(
    parameters = dict(
        tau = 1,
        lr_i = 0.004,
        target_rate = ann.Parameter(0.5),
    ),
    equations = [
        'dw/dt = lr_i*(post.r - target_rate) * pre.r',
    ]
)

NonPlastic_Synapse = ann.Synapse()

# defining some params
FAST_LR_E = 4e-3
SLOW_LR_E = 4e-4



# defining number of neurons in the network 
num_HPC_E_neuron = 40
num_HPC_I_neuron = 40

num_CTX_E_neuron = 40
num_CTX_I_neuron = 40

# defining the network with excitatory and inhibitory populations
Can_network = ann.Network()
# HPC pops
HPC_E_pop = Can_network.create(geometry=num_HPC_E_neuron, neuron=HPC_LIneuron, name='HPC_E_pop')
HPC_I_pop = Can_network.create(geometry=num_HPC_I_neuron, neuron=HPC_LIneuron, name='HPC_I_pop')

# CTX pops
CTX_E_pop = Can_network.create(geometry=num_CTX_E_neuron, neuron=HPC_LIneuron, name='CTX_E_pop')
CTX_I_pop = Can_network.create(geometry=num_CTX_I_neuron, neuron=HPC_LIneuron, name='CTX_I_pop')

# defining the synapses between excitatory and inhibitory populations, (EE, EI, IE, II), plastic are  EE, EI and II. considering II as static

## HPC connections network Rec connections

# according to complementary learning system (CLS), HPC is a fast learner for episodic memories
HPC_EE_proj = Can_network.connect(HPC_E_pop,HPC_E_pop,'exc',E_synapse)
HPC_EE_proj.connect_all_to_all(weights=0.0, allow_self_connections=True)  # Uncomment if you want to initialize HPC_EE_proj weights


HPC_EI_proj = Can_network.connect(HPC_E_pop,HPC_I_pop,'exc',E_synapse)
HPC_EI_proj.connect_all_to_all(weights=0.0)

HPC_IE_proj = Can_network.connect(HPC_I_pop,HPC_E_pop,'inh',I_synapse)
HPC_IE_proj.connect_all_to_all(weights=-0.1, allow_self_connections=True)

HPC_II_proj = Can_network.connect(HPC_I_pop,HPC_I_pop,'inh',NonPlastic_Synapse)
HPC_II_proj.connect_all_to_all(weights=-0.1, allow_self_connections=True)  # Uncomment if you want to initialize HPC_II_proj weights


## CTX connections network Rec connections
CTX_EE_proj = Can_network.connect(CTX_E_pop,CTX_E_pop,'exc',E_synapse)
CTX_EE_proj.connect_all_to_all(weights=0.0, allow_self_connections=True)  # Uncomment if you want to initialize HPC_EE_proj weights


CTX_EI_proj = Can_network.connect(CTX_E_pop,CTX_I_pop,'exc',E_synapse)
CTX_EI_proj.connect_all_to_all(weights=0.0)

CTX_IE_proj = Can_network.connect(CTX_I_pop,CTX_E_pop,'inh',I_synapse)
CTX_IE_proj.connect_all_to_all(weights=-0.1, allow_self_connections=True)

CTX_II_proj = Can_network.connect(CTX_I_pop,CTX_I_pop,'inh',NonPlastic_Synapse)
CTX_II_proj.connect_all_to_all(weights=-0.1, allow_self_connections=True)  # Uncomment if you want to initialize HPC_II_proj weights



# need to define input for each of the neurons in the population
input_current = np.zeros(num_HPC_E_neuron)

Can_network.compile(clean=True)

# # according to complementary learning system (CLS) CTX is a slow learner but better at generalizing 
# Hence need to change the learning rates for these synapses
CTX_EE_proj.lr_e = SLOW_LR_E
CTX_IE_proj.lr_i = SLOW_LR_E
CTX_EI_proj.lr_e = SLOW_LR_E

# Simulation
# step 1: a burn in period of 10 time steps
# step 2: Encoding single stimulus
# step 3: rounds of consolidation (awake + sleep cycles?)
# step 4: recall
PAT_LEN = 10
PATTERN_A = np.concat((generate_random_pattern(PAT_LEN),np.zeros(num_HPC_E_neuron-PAT_LEN)))
breakpoint()
HPC_E_pop[:].I_ext =  PATTERN_A# Set external input for first 10 neurons
HPC_I_pop[:].I_ext = 0.5 * PATTERN_A
CTX_E_pop[:].I_ext = PATTERN_A
 
# variables to store weights 
HPC_EE_W = []
HPC_EI_W = []
HPC_IE_W = []
HPC_II_W = []

CTX_EE_W = []
CTX_EI_W = []
CTX_IE_W = []
CTX_II_W = []

# setting up monitors to record the firing rates
HPC_E_act_monitor = Can_network.monitor(HPC_E_pop,'r')
HPC_I_act_monitor = Can_network.monitor(HPC_I_pop,'r') 
CTX_E_act_monitor = Can_network.monitor(HPC_E_pop,'r')
CTX_I_act_monitor = Can_network.monitor(HPC_I_pop,'r') 

# w = Can_network.monitor(HPC_EE_proj,'w')

Can_network.config(dt=0.01)
Can_network.simulate(10)
# breakpoint()

HPC_EE_W.append(HPC_EE_proj.connectivity_matrix())
HPC_EI_W.append(HPC_EI_proj.connectivity_matrix())
HPC_IE_W.append(HPC_IE_proj.connectivity_matrix())
HPC_II_W.append(HPC_II_proj.connectivity_matrix())


CTX_EE_W.append(CTX_EE_proj.connectivity_matrix())
CTX_EI_W.append(CTX_EI_proj.connectivity_matrix())
CTX_IE_W.append(CTX_IE_proj.connectivity_matrix())
CTX_II_W.append(CTX_II_proj.connectivity_matrix())

HPC_e_Act = HPC_E_act_monitor.get('r')
HPC_i_Act = HPC_I_act_monitor.get('r')

CTX_e_Act = CTX_E_act_monitor.get('r')
CTX_i_Act = CTX_I_act_monitor.get('r')

plot_activity_n_excitability_time([HPC_e_Act,CTX_e_Act],["HPC E F.R.","CTX E F.R."],fname="./plots/Sys_cons/E_activities.png",cmaps=['Blues', 'Greens'])
plot_activity_n_excitability_time([HPC_i_Act,CTX_i_Act],["HPC I F.R.","CTX E I.R."],fname="./plots/Sys_cons/I_activities.png",cmaps=['Blues', 'Greens'])

# plot_activity(hpc_e_Act,"","HPC netowrk excitatory neuron activity", "./plots/HPC_E_activity.png",c='red')
# plot_activity(hpc_i_Act,"","HPC netowrk inhibitory neuron activity", "./plots/HPC_E_activity.png",c='blue')

# before_after_weights(EE_initial_weights, HPC_EE_W, "Initial W_{EE}" ,"Final W_{EE}" ,"./plots/HPC_EE_weights.png")
# before_after_weights(EI_initial_weights, HPC_EI_W, "Initial W_{EI}" ,"Final W_{EI}" ,"./plots/HPC_EI_weights.png")
# before_after_weights(IE_initial_weights, HPC_IE_W, "Initial W_{IE}" ,"Final W_{IE}" ,"./plots/HPC_IE_weights.png")
# before_after_weights(II_initial_weights, HPC_II_W, "Initial W_{II}" ,"Final W_{II}" ,"./plots/HPC_II_weights.png")