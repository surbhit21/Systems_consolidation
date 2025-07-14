import ANNarchy as ann
import matplotlib.pyplot as plt
import numpy as np
# import os
from plotting_widget import *
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
        min_w = -1e+9,
        max_w = 1e+9,
        act_thrsh = 0.4
    ),
    equations = [
        ' tau*dw/dt =  -w * dr_e + lr_e*pre.r * post.r : min = min_w, max = max_w',
    ]
)


I_synapse = ann.Synapse(
    parameters = dict(
        tau = 1,
        lr_i = 0.004,
        target_rate = ann.Parameter(0.4),
        max_w = 1e+9,
        min_w = -1e+9
    ),
    equations = [
        'dw/dt = lr_i*(post.r - target_rate) * pre.r : min = min_w, max = max_w',
    ]
)

NonPlastic_Synapse = ann.Synapse()

# defining some params
FAST_LR_E = 4e-3
SLOW_LR_E = 1e-4



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
HPC_IE_proj.connect_all_to_all(weights=0.0)

HPC_II_proj = Can_network.connect(HPC_I_pop,HPC_I_pop,'inh',NonPlastic_Synapse)
HPC_II_proj.connect_all_to_all(weights=-0.1, allow_self_connections=True)  # Uncomment if you want to initialize HPC_II_proj weights


## CTX connections network Rec connections
CTX_EE_proj = Can_network.connect(CTX_E_pop,CTX_E_pop,'exc',E_synapse)
CTX_EE_proj.connect_all_to_all(weights=0.0, allow_self_connections=True)  # Uncomment if you want to initialize HPC_EE_proj weights


CTX_EI_proj = Can_network.connect(CTX_E_pop,CTX_I_pop,'exc',E_synapse)
CTX_EI_proj.connect_all_to_all(weights=0.0)

CTX_IE_proj = Can_network.connect(CTX_I_pop,CTX_E_pop,'inh',I_synapse)
CTX_IE_proj.connect_all_to_all(weights=0.0)

CTX_II_proj = Can_network.connect(CTX_I_pop,CTX_I_pop,'inh',NonPlastic_Synapse)
CTX_II_proj.connect_all_to_all(weights=-0.1, allow_self_connections=True)  # Uncomment if you want to initialize HPC_II_proj weights


# interregion connectivity HPC --> CTX E connections, CTX --> HPC, I connections
CTX_HPC_I_proj = Can_network.connect(CTX_I_pop,HPC_E_pop,'inh',I_synapse) 
CTX_HPC_I_proj.connect_all_to_all(weights = 0.0)

HPC_CTX_E_proj = Can_network.connect(HPC_E_pop,CTX_E_pop,'exc',E_synapse)
HPC_CTX_E_proj.connect_all_to_all(weights= 0.0)

# need to define input for each of the neurons in the population
input_current = np.zeros(num_HPC_E_neuron)

Can_network.compile(clean=True)

# # according to complementary learning system (CLS) CTX is a slow learner but better at generalizing 
# Hence need to change the learning rates for these synapses
CTX_EE_proj.lr_e = SLOW_LR_E
CTX_IE_proj.lr_i = SLOW_LR_E
CTX_EI_proj.lr_e = SLOW_LR_E


# variables to store weights 
HPC_EE_W = []
HPC_EI_W = []
HPC_IE_W = []
HPC_II_W = []

CTX_EE_W = []
CTX_EI_W = []
CTX_IE_W = []
CTX_II_W = []

CTX_HPC_I_w = []
HPC_CTX_E_w = []

HPC_E_act_monitor = Can_network.monitor(HPC_E_pop,'r')
HPC_I_act_monitor = Can_network.monitor(HPC_I_pop,'r') 
CTX_E_act_monitor = Can_network.monitor(HPC_E_pop,'r')
CTX_I_act_monitor = Can_network.monitor(HPC_I_pop,'r') 

# Simulation
# step 1: a burn in period of 10 time steps
# step 2: Encoding single stimulus
# step 3: rounds of consolidation (awake + sleep cycles?)
# step 4: recall
# setting up monitors to record the firing rates
PAT_LEN = 10
np.random.seed(1)
PATTERN_A = np.concat((generate_random_pattern(PAT_LEN),np.zeros(num_HPC_E_neuron-PAT_LEN)))

# encoding phase
HPC_E_pop[:].I_ext =  PATTERN_A# Set external input for first 10 neurons
HPC_I_pop[:].I_ext =  PATTERN_A
CTX_E_pop[:].I_ext =  PATTERN_A
CTX_I_pop[:].I_ext =  PATTERN_A

T_encode = 100
dt = 0.1
Can_network.config(dt=dt)
Can_network.simulate(T_encode)

# for i in range(7):
#     HPC_E_pop[:].I_ext =  HPC_I_pop[:].I_ext = CTX_E_pop[:].I_ext = CTX_I_pop[:].I_ext = np.zeros(num_CTX_E_neuron)
#     # rem phase
#     Can_network

# breakpoint()

HPC_EE_W.append(HPC_EE_proj.connectivity_matrix())
HPC_EI_W.append(HPC_EI_proj.connectivity_matrix())
HPC_IE_W.append(HPC_IE_proj.connectivity_matrix())
HPC_II_W.append(HPC_II_proj.connectivity_matrix())


CTX_EE_W.append(CTX_EE_proj.connectivity_matrix())
CTX_EI_W.append(CTX_EI_proj.connectivity_matrix())
CTX_IE_W.append(CTX_IE_proj.connectivity_matrix())
CTX_II_W.append(CTX_II_proj.connectivity_matrix())

CTX_HPC_I_w.append(CTX_HPC_I_proj.connectivity_matrix())
HPC_CTX_E_w.append(HPC_CTX_E_proj.connectivity_matrix())

# no_input_time = 10
# no_input = np.zeros(num_HPC_E_neuron)

# HPC_E_pop[:].I_ext =  no_input# Set external input for first
# HPC_I_pop[:].I_ext =  no_input
# CTX_E_pop[:].I_ext =  no_input
# CTX_I_pop[:].I_ext =  no_input
# Can_network.simulate(no_input_time)

# Recall Phase
T_recall = 20
partial_pattern = np.zeros(num_HPC_E_neuron)
per = 0.8
PAT_LEN = 10
partial_pattern[:int(PAT_LEN*per)] = PATTERN_A[:int(PAT_LEN*per)]
HPC_E_pop[:].I_ext =  partial_pattern# Set external input for first
HPC_I_pop[:].I_ext =  partial_pattern
CTX_E_pop[:].I_ext =  partial_pattern
CTX_I_pop[:].I_ext =  partial_pattern

Can_network.simulate(T_recall)






# Plotting everything
HPC_e_Act = HPC_E_act_monitor.get('r')
HPC_i_Act = HPC_I_act_monitor.get('r')

CTX_e_Act = CTX_E_act_monitor.get('r')
CTX_i_Act = CTX_I_act_monitor.get('r')

encoding_HPC_activity = HPC_e_Act[:int(T_encode//dt),:]
recall_HPC_activity = HPC_e_Act[-int(T_recall//dt):,:]

encoding_ctx_activity = CTX_e_Act[:int(T_encode//dt),:]
recall_ctx_activity = CTX_e_Act[-int(T_recall//dt):,:]

plot_activity_n_excitability_time([HPC_e_Act.T,CTX_e_Act.T],["HPC E F.R.","CTX E F.R."],fname="./plots/Sys_cons/E_activities.png",cmaps=['Blues', 'Greens'])
plot_activity_n_excitability_time([HPC_i_Act.T,CTX_i_Act.T],["HPC I F.R.","CTX E I.R."],fname="./plots/Sys_cons/I_activities.png",cmaps=['Blues', 'Greens'])

breakpoint()
plot_avg_activity([encoding_HPC_activity.T,encoding_ctx_activity.T,recall_HPC_activity.T,recall_ctx_activity.T],["HPC E F.R.","CTX E F.R."],fname="./plots/Sys_cons/Encode_activities.png",cmaps=['red', 'red', 'blue', 'blue'])
# plot_avg_activity([recall_HPC_activity.T,recall_ctx_activity.T],["HPC E F.R.","CTX E F.R."],fname="./plots/Sys_cons/recall_activities.png",cmaps=['gray', 'gray'])

# # plot_activity(hpc_e_Act,"","HPC netowrk excitatory neuron activity", "./plots/HPC_E_activity.png",c='red')
# # plot_activity(hpc_i_Act,"","HPC netowrk inhibitory neuron activity", "./plots/HPC_E_activity.png",c='blue')

# plot_weights_over_time([HPC_EE_W[0], CTX_EE_W[0]],[ "HPC W_{EE}" ,"CTX W_{EE}"] ,"./plots/Sys_cons/EE_weights.png")
# plot_weights_over_time([HPC_EI_W[0], CTX_EI_W[0]], ["HPC W_{EI}" ,"CTX W_{EI}" ],"./plots/Sys_cons/EI_weights.png")
# plot_weights_over_time([HPC_IE_W[0], CTX_IE_W[0]], ["HPC W_{IE}" ,"CTX W_{IE}"] ,"./plots/Sys_cons/IE_weights.png")
# plot_weights_over_time([HPC_II_W[0], CTX_II_W[0]], ["HPC W_{II}" ,"CTX W_{II}"] ,"./plots/Sys_cons/II_weights.png")

# plot_weights_over_time([CTX_HPC_I_w[0], HPC_CTX_E_w[0]], ["CTX => HPC: w" ,"HPC => CTX: w"] ,"./plots/Sys_cons/cross_connections.png")
