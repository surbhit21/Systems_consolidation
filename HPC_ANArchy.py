import ANNarchy as ann
import numpy as np
import matplotlib.pyplot as plt
HPC_LIneuron = ann.Neuron(
    parameters = dict(
        tau = 10.0,
        baseline = 0,
        I_ext = ann.Parameter(0.1),
    ),
    equations = [
        'tau * dr/dt  + r = baseline + tanh(sum(exc) + I_ext)'
    ]
) 

HPC_synapse = ann.Synapse(
    parameters = dict(
        tau = 10.0,
        learning_rater = 1e-3,
        decay_rate = ann.Parameter(0.1),
    ),
    equations = [
        'tau * dw/dt = - w * decay_rate + learning_rater*pre.r * post.r',
    ]
)

HPC_network = ann.Network()

pop1 = HPC_network.create(geometry=10, neuron=HPC_LIneuron, name='pop1')
pop1[5:].I_ext = 10
proj = HPC_network.connect(pop1,pop1,'exec',HPC_synapse)

proj.connect_all_to_all(weights=0.1)

HPC_network.compile(clean=True)


m = HPC_network.monitor(pop1,'r') 
w = HPC_network.monitor(proj,'w')

HPC_network.config(dt=0.1)
HPC_network.simulate(100)
# breakpoint()

frs = m.get('r')
weigts = w.get('w')
# breakpoint()
plt.figure(figsize=(10, 5))
plt.subplot(121)
# plt.title('Outputs of the Rate Network over Time')
plt.plot(frs[:,0:10])
plt.xlabel('Time Steps')
plt.ylabel('firing rates')
plt.legend([f'Neuron {i}' for i in range(10)])
plt.title('Firing rates')
plt.subplot(122)
plt.plot(weigts[:,0,0], '-')
plt.plot(weigts[:,1,0], '.')
plt.title('Weights')
plt.show()

