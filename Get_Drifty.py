import ANNarchy as ann
import matplotlib.pyplot as plt
import numpy as np
from plotting_widget import *
from Utilities import get_active_neurons,ensamble_overlap
LIneuron = ann.Neuron(
    parameters = dict(
        tau = 50,
        input_i = ann.Parameter(0.0,locality="local"),  # input current
        I0 = 5,
        I1 = 1,
        I2 = 0.01,
        Epsi_i = ann.Parameter(0.0,locality="local"),  # Epsi_i parameter
        
    ),
    equations = [
        'tau * dx/dt  + x = pos(input_i - I0 - I1 * norm1(x) - I2 * norm2(x) + Epsi_i + sum(exc))  ',  # reset small values to zero
        'r = if x < 1e-5: 0 else: x',  # firing rate
        'ex = Epsi_i',
        'inp = input_i'
        
    ]
) 

OPneuron = ann.Neuron(
    parameters = dict(
        tau = 100,
        min_fr = 0,
        max_fr = 100,
        ),
    equations = [
        'r = sum(exc) : min = min_fr,max = max_fr ',
        # 'r = if x < 1e-2: 0 else : x : min = min_fr,max = max_fr '
    ]
) 

# tau * dr/dt  + r = max(0, Input_i - I0 - I1 * sum(r) - I2 * sum(r*r) + Epsi_i + sum(exc))

E_synapse = ann.Synapse(
    parameters = dict(
        tau_w = 1000,
        lr = 1,
        tau_decay = 1000,
        max_weight = 0.4,
        min_weight = 0.0,
    ),
    equations = [
        'dw/dt = (lr*(pre.r * post.r)/tau_w - w/tau_decay) : min = min_weight, max = max_weight'
        # 'Bounded(x) = np.clip(x, 0, 1)'
    ]
)

# HomeoE_synapse = ann.Synapse(
#     parameters = dict(
#         tau_w = 200,
#         lr = 1,
#         tau_decay = 1000,
#         max_weight = 1,
#         min_weight = 0,
#         total_weight = 0
#     ),
#     equations = [
#         'dw/dt = (lr*(1-total_weight)*(pre.r * post.r)/tau_w - w/tau_decay)'
#         # 'Bounded(x) = np.clip(x, 0, 1)'
#     ]
# )

np.random.seed(1)
num_HPC_E_neuron = 70
# normal distribution parameters for Epsi_i
mu = 0
sigma = 1
orig_exitability = np.abs(np.random.uniform(mu, sigma, num_HPC_E_neuron))
E = 3.5 # Epsi_i increase factor for neurons

# encoding time duration
T_encode = 200


# Offline simulation parameters
Nrep = 3 # number of repetitions
T = 200 # duration of repetitions
IR = 100 # Inter-repetition interval
num_days = 5 # number of offline days in the simulation
ID = 1000 #inter-day delay

# recall time duration
T_recall = 200


delta = 50 # input current
theta = 2 #threshold firing rate for active neurons
c = 1 #cap pn recurrent weights

del_t = 1 # time step


net = ann.Network()
E_pop = net.create(geometry=num_HPC_E_neuron, neuron=LIneuron, name='E_pop')
EE_proj = net.connect(E_pop,E_pop,'exc',E_synapse)
EE_proj.connect_all_to_all(weights=0,allow_self_connections=True)

output_neuron = net.create(geometry=1, neuron=OPneuron, name='op_pop')
Eop_proj = net.connect(E_pop,output_neuron,'exc',E_synapse)
Eop_proj.connect_all_to_all(weights=0.01)

net.compile(clean=True)

Eop_proj.lr = 0.1
Eop_proj.tau_decay = 1000
Eop_proj.tau_w = 400
# Eop_proj.max_weight = 1
# Eop_proj.min_weight = -1

# E_pop[10:].input_i = np.array(40*[0])
# E_pop[10:].Epsi_i = np.array(40*[0])

# setting time step
net.config(dt=del_t)


# setting the monitor for the excitatory population
E_act_monitor = net.monitor(E_pop,'r')
E_Epsi_i_monitor = net.monitor(E_pop,'ex')
E_input_monitor = net.monitor(E_pop,'inp')
op_act_monitor = net.monitor(output_neuron,'r')
op_weights_monitor = net.monitor(Eop_proj, 'w')
weights = []
op_weights = []
activity_vector = []

# ENCODING PHASE
E_pop[:].Epsi_i = orig_exitability
# increasing excitability of first 10 neurons
E_pop[0:10].Epsi_i += E
# setting input to each neurn
E_pop[:].input_i = np.array([delta]*num_HPC_E_neuron)
net.simulate(T_encode)
encoding_ensamble = get_active_neurons(E_pop.r,theta)
activity_vector.append(E_pop.r)
weights.append(EE_proj.connectivity_matrix())
op_weights.append(Eop_proj.connectivity_matrix())

# OFFLINE PHASE
offline_ensamble = []
for day in range(1,num_days+1):
    # setting excitability for different set of neurons each day
    E_pop[:].Epsi_i = orig_exitability #np.abs(np.random.uniform(mu, sigma, num_HPC_E_neuron))
    E_pop[(day)*10:(day+1)*10].Epsi_i += E
    for j in range(Nrep):
        # seed = j
        # setting a random seed
        # np.random.seed(j)
        # stim_phase 
        # input is set for all neurons in the population
        E_pop[:].input_i = np.array([delta]*num_HPC_E_neuron)
        # exitability is drawn from a uniform distri
        # now based on day the exicitability is increased by a factor of E for some neurons
        # if day == 1:
        #     EE_proj.lr = 1e-4
        # else:
        #     EE_proj.lr = 1
        # run the simulation for T seconds
        net.simulate(T)
        # need to RUN step by step for 
        # for i in range(0,T):
            # w_tot = np.sum(Eop_proj.connectivity_matrix())
            # print(w_tot)
            # Eop_proj.total_weight = w_tot
        #     net.step()
        # breakpoint()
        # print(E_pop[1].r)
        # post-stim (pause) phase 
        # input is set to 0 for all neurons in the population
        E_pop[:].input_i = np.array([0]*num_HPC_E_neuron)
        # Epsi_i remains elevated for the same neurons
        net.simulate(IR)
        # for i in range(0,IR):
            # w_tot = np.sum(Eop_proj.connectivity_matrix())
            # # print(w_tot)
            # Eop_proj.total_weight = w_tot
            # net.step()
    # breakpoint()
    
        # breakpoint()
    offline_ensamble.append(get_active_neurons(E_pop.r,theta))
    activity_vector.append(E_pop.r)
    weights.append(EE_proj.connectivity_matrix())
    op_weights.append(Eop_proj.connectivity_matrix())
    net.simulate(ID)
    # for i in range(0,ID):
        # w_tot = np.sum(Eop_proj.connectivity_matrix())
        # # print(w_tot)
        # Eop_proj.total_weight = w_tot
        # net.step()

# plot_activity(E_act_monitor,'r',"Neuron activity", "./plots/delamare_2024_F1B.png")
# plot_activity(E_Epsi_i_monitor,'ex',"HPC netowrk excitatory neuron activity", "./plots/delamare_2024_F1A.png")

# Time for recall
recall_ensamble = []
E_pop[:].Epsi_i = orig_exitability
# increasing excitability of first 10 neurons
E_pop[-10:].Epsi_i += E
# setting input to each neurn
E_pop[:].input_i = np.array([delta]*num_HPC_E_neuron)
net.simulate(T_recall)
recall_ensamble = get_active_neurons(E_pop.r,theta)
activity_vector.append(E_pop.r)
weights.append(EE_proj.connectivity_matrix())
op_weights.append(Eop_proj.connectivity_matrix())


breakpoint()
rs = E_act_monitor.get('r').T
oprs = op_act_monitor.get('r').T
# rs[rs < theta] = 0  # Ensure firing rates are between 0 and 1
exs = E_Epsi_i_monitor.get('ex').T
inps =E_input_monitor.get('inp').T
op_w = op_weights_monitor.get('w')
op_w = np.array(op_w[:,0,:].T,dtype='float')
# overlapping_ensambles = ensamble_overlap(encoding_ensamble,offline_ensamble)
# op_weights = np.array(op_weights,dtype='float')[:,0,:]
rs = np.where(rs > theta, rs, 0)
# breakpoint()
E_name = str(E).replace('.','_')
I_name = str(delta).replace('.','_')
plot_activity_n_excitability_time([rs,exs,inps],
                       titles=['Neuronal Activity',
                                'Neuronal Excitability',
                                'Input current'],
                       fname="./plots/E_{0}_{1}/delamare_2024_F1AB.png".format(E_name,I_name),
                       cmaps=['Blues', 'Greens', 'Reds'])
plot_activity_n_excitability_time([oprs,op_w],
                                  titles=['Neuronal Activity','Synaptic weights'],
                                  fname="./plots/E_{0}_{1}/delamare_2024_F3AB.png".format(E_name,I_name),
                                  cmaps=['Blues', 'gray_r', 'Reds'])
plot_activity(oprs[0],'','Output neuron FR',fname="./plots/E_{0}_{1}/delamare_2024_F3B.png".format(E_name,I_name),c='r')
plot_weights_over_time(weights,
                       titles= ["Encoding"]+[f'Day {i+1}' for i in range(num_days)] + ["Recall"],
                       fname="./plots/E_{0}_{1}/delamare_2024_F1C.png".format(E_name,I_name),
                       cmaps='gray_r')
# # plot_weights_over_time(op_weights,
# #                        titles=[f'Day {i+1}' for i in range(len(weights))],
# #                        fname="./plots/E_{0}_{1}/delamare_2024_F3C.png".format(E_name,I_name),
# #                        cmaps='gray_r')
# correlation_plot
activity_vector = np.array(activity_vector)
breakpoint()
plot_row_correlations(activity_vector[0],activity_vector[1:-1], fname="./plots/E_{0}_{1}/encoding_corr.png".format(E_name,I_name), use_bar_plot=True)
plot_row_correlations(activity_vector[-1],activity_vector[1:-1], fname="./plots/E_{0}_{1}/Recall_corr.png".format(E_name,I_name), ref_index=-1, use_bar_plot=True)
plot_consecutive_day_correlation(np.array(activity_vector), fname="./plots/E_{0}_{1}/dayN_N_1_corr.png".format(E_name,I_name))
plot_rowwise_com(op_weights.T,num_days,fname="./plots/E_{0}_{1}/delamare_2024_F3D.png".format(E_name,I_name))