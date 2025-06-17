import ANNarchy as ann
import matplotlib.pyplot as plt
import numpy as np
from plaotting_widget import before_after_weights, plot_activity,plot_weights_over_time


LIneuron = ann.Neuron(
    parameters = dict(
        tau = 20,
        input_i = ann.Parameter(15),
        I0 = 12,
        I1 = 0.5*50,
        I2 = 0.05*50,
        excitabilty = ann.Parameter(0.0),
    ),
    equations = [
        'tau * dr/dt  + r = max(0, input_i - I0 - I1 * norm1(r) - I2 * norm2(r) + excitabilty + sum(exc))',
        'ex = excitabilty',
        
    ]
) 

# tau * dr/dt  + r = max(0, Input_i - I0 - I1 * sum(r) - I2 * sum(r*r) + excitabilty + sum(exc))

E_synapse = ann.Synapse(
    parameters = dict(
        tau_w = 1000,
        tau_decay = 1000,
        max_weight = 3.0,
        min_weight = 0.0,
    ),
    equations = [
        'dw = ((pre.r * post.r)/tau_w - w/tau_decay)*dt : min = min_weight, max = max_weight'
        # 'Bounded(x) = np.clip(x, 0, 1)'
    ]
)

num_HPC_E_neuron = 50

net = ann.Network()
E_pop = net.create(geometry=num_HPC_E_neuron, neuron=LIneuron, name='E_pop')
EE_proj = net.connect(E_pop,E_pop,'exc',E_synapse)
EE_proj.connect_all_to_all(weights=0.0)

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

delta = 15 # input current
theta = 5 #threshold firing rate for active neurons
c = 1 #cap pn recurrent weights
E = 1.5 # excitability increase factor for neurons

# normal distribution parameters for excitability
mu = 0
sigma = 1
net.config(dt=del_t)

# setting the monitor for the excitatory population

E_act_monitor = net.monitor(E_pop,'r')
E_excitability_monitor = net.monitor(E_pop,'ex')
weights = []
weights.append(EE_proj.connectivity_matrix())
for i in range(num_days):
    
    for j in range(Nrep):
        seed = j
        # setting a random seed
        np.random.seed(seed)
        # stim_phase 
        # input is set for all neurons in the population
        E_pop[:].input_i = np.array(num_HPC_E_neuron*[delta])
        # exitability is drawn from a uniform distribution
        E_pop[:].excitabilty = np.abs(np.random.uniform(mu, sigma, num_HPC_E_neuron))
        # now based on day the exicitability is increased by a factor of E for some neurons
        E_pop[(i+1)*10:(i+1)*10+10].excitabilty += E
        # run the simulation for T seconds
        net.simulate(T)
        # post-stim (pause) phase 
        # input is set to 0 for all neurons in the population
        E_pop[:].input_i = np.array(num_HPC_E_neuron*[0])
        # excitability remains elevated for the same neurons
        net.simulate(IR)
    net.simulate(ID)
    weights.append(EE_proj.connectivity_matrix())

# plot_activity(E_act_monitor,'r',"Neuron activity", "./plots/delamare_2024_F1B.png")
# plot_activity(E_excitability_monitor,'ex',"HPC netowrk excitatory neuron activity", "./plots/delamare_2024_F1A.png")



before_after_weights(E_act_monitor.get('r').T,E_excitability_monitor.get('ex').T,"Neuronal Activity","Neuronal Excitability","./plots/delamare_2024_F1AB.png",cmaps= 'Greens')
# before_after_weights(weights[0],weights[1],"day 0-1 ","./plots/delamare_2024_F1d1.png",cmaps= 'gray')
# before_after_weights(weights[1],weights[2],"day 1-2 ","./plots/delamare_2024_F1d1.png",cmaps= 'gray')
# before_after_weights(weights[3],weights[4],"days3-4","./plots/delamare_2024_F1d2.png",cmaps= 'gray')
plot_weights_over_time(weights,
                       titles=[f'Day {i+1}' for i in range(len(weights))],
                       fname="./plots/delamare_2024_F1C.png",
                       cmaps='gray')
breakpoint()
