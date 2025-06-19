import ANNarchy as ann
import matplotlib.pyplot as plt
import numpy as np
import os
from plaotting_widget import before_after_weights, plot_activity

HPC_LIneuron = ann.Neuron(
    parameters = dict(
        tau = 10.0,
        baseline = 0,
        I_ext = ann.Parameter(0.1),
        excitabilty = 1.0,
    ),
    equations = [
        'tau * dr/dt  + r = baseline + tanh(sum(exc) + sum(inh) + I_ext)'
    ]
) 

HPC_E_synapse = ann.Synapse(
    parameters = dict(
        tau = 1,
        lr_e = 0.004,
        dr_e = ann.Parameter(0),
    ),
    equations = [
        'tau * dw/dt = -w * dr_e + lr_e*pre.r * post.r',
    ]
)

HPC_I_synapse = ann.Synapse(
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

# defining number of neurons in the networ
num_HPC_E_neuron = 40
num_HPC_I_neuron = 40

# defining the network with excitatory and inhibitory populations
HPC_network = ann.Network()
HPC_E_pop = HPC_network.create(geometry=num_HPC_E_neuron, neuron=HPC_LIneuron, name='HPC_E_pop')
HPC_I_pop = HPC_network.create(geometry=num_HPC_I_neuron, neuron=HPC_LIneuron, name='HPC_I_pop')

# dfining the synapses between excitatory and inhibitory populations, (EE, EI, IE, II), plastic are  EE, EI and II. considering II as static
EE_proj = HPC_network.connect(HPC_E_pop,HPC_E_pop,'exc',HPC_E_synapse)
EE_proj.connect_all_to_all(weights=0.0)

EI_proj = HPC_network.connect(HPC_E_pop,HPC_I_pop,'exc',HPC_E_synapse)
EI_proj.connect_all_to_all(weights=0.0)

IE_proj = HPC_network.connect(HPC_I_pop,HPC_E_pop,'inh',HPC_I_synapse)
IE_proj.connect_all_to_all(weights=0.0)

II_proj = HPC_network.connect(HPC_I_pop,HPC_I_pop,'inh',NonPlastic_Synapse)
II_proj.connect_all_to_all(weights=-0.1)  # Uncomment if you want to initialize II_proj weights


HPC_E_pop[:10].I_ext = np.array([1,-1,1,1,-1,-1,-1,1,1,1] ) # Set external input for first 10 neurons
HPC_I_pop[:10].I_ext = 0.5*np.array([1,-1,1,1,-1,-1,-1,1,1,1])


HPC_network.compile(clean=True)


EE_initial_weights = EE_proj.connectivity_matrix()
EI_initial_weights = EI_proj.connectivity_matrix()
IE_initial_weights = IE_proj.connectivity_matrix()
II_initial_weights = II_proj.connectivity_matrix()


HPC_E_act_monitor = HPC_network.monitor(HPC_E_pop,'r')
HPC_I_act_monitor = HPC_network.monitor(HPC_I_pop,'r') 
# w = HPC_network.monitor(EE_proj,'w')

HPC_network.config(dt=0.01)
HPC_network.simulate(10)
# breakpoint()

EE_final_weights = EE_proj.connectivity_matrix()
EI_final_weights = EI_proj.connectivity_matrix()
IE_final_weights = IE_proj.connectivity_matrix()
II_final_weights = II_proj.connectivity_matrix()


plot_activity(HPC_E_act_monitor,'r',"HPC netowrk excitatory neuron activity", "./plots/HPC_E_activity.png")
plot_activity(HPC_I_act_monitor,'r',"HPC netowrk inhibitory neuron activity", "./plots/HPC_E_activity.png")

before_after_weights(EE_initial_weights, EE_final_weights, "Initial W_{EE}" ,"Final W_{EE}" ,"./plots/HPC_EE_weights.png")
before_after_weights(EI_initial_weights, EI_final_weights, "Initial W_{EI}" ,"Final W_{EI}" ,"./plots/HPC_EI_weights.png")
before_after_weights(IE_initial_weights, IE_final_weights, "Initial W_{IE}" ,"Final W_{IE}" ,"./plots/HPC_IE_weights.png")
before_after_weights(II_initial_weights, II_final_weights, "Initial W_{II}" ,"Final W_{II}" ,"./plots/HPC_II_weights.png")