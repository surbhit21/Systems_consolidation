import ANNarchy as ann
import matplotlib.pyplot as plt
import numpy as np
from plaotting_widget import before_after_weights, plot_activity,plot_weights_over_time,plot_row_correlations


LIneuron = ann.Neuron(
    parameters = dict(
        tau = 50,
        input_i = ann.Parameter(0.0,locality="local"),  # input current
        I0 = 5,
        I1 = 1,
        I2 = 0.01,
        excitabilty = ann.Parameter(0.0,locality="local"),  # excitability parameter
    ),
    equations = [
        'tau * dx/dt  + x = pos(input_i - I0 - I1 * norm1(x) - I2 * norm2(x) + excitabilty + sum(exc))  ',  # reset small values to zero
        'r = if x < 1e-5: 0 else: x',  # firing rate
        'ex = excitabilty',
        'inp = input_i'
        
    ]
) 

# tau * dr/dt  + r = max(0, Input_i - I0 - I1 * sum(r) - I2 * sum(r*r) + excitabilty + sum(exc))

E_synapse = ann.Synapse(
    parameters = dict(
        tau_w = 1000,
        tau_decay = 1000,
        max_weight = 0.4,
        min_weight = 0.0,
    ),
    equations = [
        'dw/dt = ((pre.r * post.r)/tau_w - w/tau_decay) : min = min_weight, max = max_weight'
        # 'Bounded(x) = np.clip(x, 0, 1)'
    ]
)

num_HPC_E_neuron = 50

net = ann.Network()
E_pop = net.create(geometry=num_HPC_E_neuron, neuron=LIneuron, name='E_pop')
EE_proj = net.connect(E_pop,E_pop,'exc',E_synapse)
EE_proj.connect_all_to_all(weights=0.0,allow_self_connections=True)

# E_pop[10:].input_i = np.array(40*[0])
# E_pop[10:].excitabilty = np.array(40*[0])

net.compile(clean=True)

# Simulation parameters
Nrep = 10 # number of repetitions
T = 100 # duration of repetitions
IR = 100 # Inter-repetition interval
del_t = 1 # time step


num_days = 4 # number of days in the simulation
ID = 1000 #inter-day delay

delta = 35 # input current
theta = 5 #threshold firing rate for active neurons
c = 1 #cap pn recurrent weights
E = 1.5 # excitability increase factor for neurons
thr = 1
# normal distribution parameters for excitability
mu = 0
sigma = 1
net.config(dt=del_t)

# setting the monitor for the excitatory population

E_act_monitor = net.monitor(E_pop,'r')
E_excitability_monitor = net.monitor(E_pop,'ex')
E_input_monitor = net.monitor(E_pop,'inp')
weights = []
# weights.append(EE_proj.connectivity_matrix())
net.simulate(ID//2)
# E_pop[:].excitabilty = np.array([0]*num_HPC_E_neuron)
orig_exitability = np.abs(np.random.uniform(mu, sigma, num_HPC_E_neuron))
np.random.seed(2)
E_pop[:].excitabilty = orig_exitability
activity_vector = []
for i in range(num_days):
    activity_vector.append([])
    for j in range(Nrep):
        seed = j
        # setting a random seed
        np.random.seed(seed)
        # stim_phase 
        # input is set for all neurons in the population
        E_pop[:].input_i = np.array(num_HPC_E_neuron*[delta])
        # exitability is drawn from a uniform distri
        # now based on day the exicitability is increased by a factor of E for some neurons
        E_pop[:].excitabilty = orig_exitability
        E_pop[(i+1)*10:(i+1)*10+10].excitabilty += E
        # run the simulation for T seconds
        net.simulate(T)
        # print(E_pop[1].r)
        # post-stim (pause) phase 
        # input is set to 0 for all neurons in the population
        E_pop[:].input_i = np.zeros(num_HPC_E_neuron)
        # excitability remains elevated for the same neurons
        net.simulate(IR)
        activity_vector[i].append(E_pop[:].r)
    activity_vector[i] = np.array(activity_vector[i])
    net.simulate(ID)
    weights.append(EE_proj.connectivity_matrix())

# plot_activity(E_act_monitor,'r',"Neuron activity", "./plots/delamare_2024_F1B.png")
# plot_activity(E_excitability_monitor,'ex',"HPC netowrk excitatory neuron activity", "./plots/delamare_2024_F1A.png")


rs = E_act_monitor.get('r').T
# rs[rs < thr] = 0  # Ensure firing rates are between 0 and 1
exs = E_excitability_monitor.get('ex').T
inps =E_input_monitor.get('inp').T
# before_after_weights(rs,exs,"Neuronal Activity","Neuronal Excitability","./plots/delamare_2024_F1AB.png",cmaps= 'Greens')
# before_after_weights(rs,inps,"Neuronal Activity","Input","./plots/delamare_2024_F1F.png",cmaps= 'Greens')
# before_after_weights(weights[0],weights[1],"day 0-1 ","./plots/delamare_2024_F1d1.png",cmaps= 'gray')
# before_after_weights(weights[1],weights[2],"day 1-2 ","./plots/delamare_2024_F1d1.png",cmaps= 'gray')
# before_after_weights(weights[3],weights[4],"days3-4","./plots/delamare_2024_F1d2.png",cmaps= 'gray')
rs = np.where(rs > thr, rs, 0)
# breakpoint()
plot_weights_over_time([rs,exs],
                       titles=['Neuronal Activity',
                                'Neuronal Excitability'],
                       fname="./plots/delamare_2024_F1AB.png",
                       cmaps=['Blues', 'Greens', 'Reds'])
plot_weights_over_time(weights,
                       titles=[f'Day {i+1}' for i in range(len(weights))],
                       fname="./plots/delamare_2024_F1C.png",
                       cmaps='gray_r')
plot_row_correlations(np.array(activity_vector), fname="./plots/delamare_2024_F2A.png")