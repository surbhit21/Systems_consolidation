import matplotlib.pyplot as plt
import numpy as np
import os
from plotting_widget import *
import torch 
import torch.nn.functional as F

class twolayer_FF:
    def __init__(self, n_inp, n_neurons,baseline_e,tau=10.0, dt=1.0, act=torch.relu, lr=8e-2, decay_r=1e-2, threshold=0, I0=1, I1=0.05, I2=0.001):
        self.n_neurons = n_neurons
        self.tau = tau  # rate constant 
        self.dt = dt  # discretization step
        self.exc = torch.abs(torch.rand(n_neurons))  # excitability 
        self.act = act  # activation function
        self.lr = lr  # learning rate for synaptic weights
        self.decay_r = decay_r  # decay rate for synaptic weights
        self.I0 = I0
        self.I1 = I1
        self.I2 = I2
        self.excitability = baseline_e
        self.threshold = threshold
        
        # Initialize random input weights
        self.input_w = torch.abs(torch.randn(n_inp, n_neurons)) / n_inp**0.5
        
        
        # Zero initial rate state
        self.rates = torch.zeros(n_neurons)
    
    def step(self, input_FR):
        """
        Perform one timestep of rate dynamics:
        input_vector: shape [n_neurons]
        """
        # calculating the input to the RNN
        input_vector = input_FR @ self.input_w 

        # blanket inhibition to the RNN
        I_inhib = self.I0 + self.I1 * torch.sum(self.rates) + self.I2 + torch.sum(self.rates**2)

        # total input to the RNN
        input_current =  input_vector + self.exc - I_inhib + self.excitability

        # rate change as the nonlinear ODE
        dr_dt = (-self.rates +  self.act(input_current)) / self.tau

        self.rates += (dr_dt * self.dt)
    
        post_mask = (self.rates > self.threshold).float()
        
        # hebbian plasticity in RNN weights
        hebbian_dw = self.lr * torch.outer(input_FR, self.rates*post_mask) * self.dt

        decay = self.decay_r * self.input_w * self.dt
        # hebbian plasticity in input weights
        self.input_w += hebbian_dw - decay

        self.input_w = torch.clamp(self.input_w, 0.0, 1.0)  # Ensure weights are non-negative
        return self.rates.detach().clone()

n = 140
e = 3.5
n_drifti ,ni_FC = 40,10 #(40 drify and 10 contextual neurons)
ni = n_drifti + ni_FC
act_threshold = 1
base_e = torch.abs(torch.normal(mean=0.0, std=1.0, size=(n,))) * 0.01
nn = twolayer_FF(ni,n,baseline_e=base_e,tau=50.0,dt = 0.1,lr = 0.05,threshold=0)

timesteps = 200
input = torch.abs(torch.normal(mean=0.0, std=1.0, size=(ni,)))
FC_input = input.clone()
FC_input[n_drifti:] += 10
nn.exc[:20] += e
nonFC_input = input
nonFC_input[:n_drifti] += 0
# nonFC_input[10:] += 1
FR_history = []
EX_history = []
rec_weights = []
ff_weights = []

# rec_weights.append(nn.W.detach().clone().numpy())
# ff_weights.append(nn.input_w.detach().clone().numpy())

Input_to_network = FC_input
for t in range(timesteps):
    next_FR = nn.step(Input_to_network)
    frs = (next_FR.numpy() > act_threshold)
    FR_history.append(next_FR.numpy()*frs)
    EX_history.append(nn.exc.detach().clone().numpy())

# rec_weights.append(nn.W.detach().clone().numpy())
ff_weights.append(nn.input_w.detach().clone().numpy())

input_history = np.tile(Input_to_network.cpu().numpy(), (timesteps, 1))


# single offline sessio
t_off = 200
Nrep = 1
binary_array = torch.randint(0, 2, (10,))
print(binary_array)
n_off_days = 5
for day in range(n_off_days):
    torch.manual_seed(day)  # Set seed for reproducibility
    nonFC_input = torch.abs(torch.normal(mean=0.0, std=1.0, size=(ni,))) 
    nn.exc[(day)*20:(day+1)*20] -= e
    nn.exc[(day+1)*20:(day+2)*20] += e
    for i in range(Nrep):
        if binary_array[i] == 2:
            Input_to_network = FC_input
        else:
            Input_to_network = nonFC_input
        for t in range(t_off):
            next_FR = nn.step(nonFC_input)
            frs = (next_FR.numpy() > act_threshold)
            FR_history.append(next_FR.numpy()*frs)
            EX_history.append(nn.exc.detach().clone().numpy())
        input_at_t = np.tile(Input_to_network, (t_off, 1))
        # breakpoint()
        input_history = np.concatenate((input_history,input_at_t))
        # input_history = np.concatenate([input_history, input_at_t[np.newaxis, :]], axis=0)
    ff_weights.append(nn.input_w.detach().clone().numpy())

# rec_weights.append(nn.W.detach().clone().numpy())
# t_recall = 200
# for t in range(t_recall):
#     nn.exc[(n_off_days)*20:(n_off_days+1)*20] -= e
#     nn.exc[:-20] += e
#     next_FR = nn.step(FC_input)
#     frs = (next_FR.numpy() > act_threshold)
#     FR_history.append(next_FR.numpy()*frs)
#     EX_history.append(nn.exc.detach().clone().numpy())

# ff_weights.append(nn.input_w.detach().clone().numpy())
FR_history = np.stack(FR_history)
EX_history = np.stack(EX_history)
breakpoint()

plot_activity_n_excitability_time([FR_history.T,EX_history.T,input_history.T],
                       titles=['Neuronal Activity',
                                "Neuronal Excitability",
                                "Input FR"],
                       fname="./plots/Reimagined/Activity_n_excitability.png",
                       cmaps=['Blues', 'Greens',"Reds"])
# plot_weights_over_time(rec_weights,
#                        titles= ["before","after"] ,
#                        fname="./plots/Reimagined/Rec_w.png",
#                        cmaps='gray_r')
plot_weights_over_time(ff_weights,
                       titles= ["FC","Of1","Of2","Of3","Of4","Of5","Recalls"] ,
                       fname="./plots/Reimagined/FF_w.png",
                       cmaps='gray_r')