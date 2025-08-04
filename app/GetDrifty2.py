import matplotlib.pyplot as plt
import numpy as np
import os
from plotting_widget import *
import torch 
import torch.nn.functional as F

class twolayermodel:
    def __init__(self,n_inp,n_neurons, tau=10.0, dt=1.0, act=torch.relu,lr = 1e-3,decay_r = 1e-4,threshold = 0,I0 = 10, I1 = 0.5, I2 = 0.01):
        self.n_neurons = n_neurons
        self.tau = tau # rate constant 
        self.dt = dt # discretization step
        self.exc = torch.abs(torch.rand(n_neurons)) # excitability 
        self.act = act # activation function
        self.lr = lr # learning rate for synaptic weights
        self.decay_r = decay_r #decay rate for synaptic weights
        self.I0 = I0
        self.I1 = I1
        self.I2 = I2
        self.threshold = threshold
        # Initialize random recurrent weights
        self.input_w = torch.abs(torch.randn(n_inp,n_neurons)) / n_inp**0.5
        self.W = torch.abs(torch.zeros(n_neurons, n_neurons)) / n_neurons**0.5
        # Zero initial rate state
        self.rates = torch.zeros(n_neurons)
    
    def step(self,input_FR):
        """
        Perform one timestep of rate dynamics:
        input_vector: shape [n_neurons]
        """
        # calculating the input to the RNN
        input_vector = input_FR @ self.input_w 

        # blancket inhibition to the RNN
        I_inhib = self.I0 + self.I1 * torch.sum(self.rates) + self.I2 + torch.sum(self.rates**2)

        # total input to the RNN
        input_current = self.W @ self.rates + input_vector + self.exc - I_inhib

        # rate change as the nonlinear ODE
        dr_dt = (-self.rates + self.act(input_current)) / self.tau

        self.rates += (dr_dt*self.dt)
    
        post_mask = (self.rates > self.threshold).float()
        # hebbian plasticity in RNN weights
        hebbian_dw = self.lr * torch.outer(self.rates*post_mask,self.rates) * self.dt

        # habbian plasticity in input weights
        hebbian_d_inputw = self.lr * torch.outer(self.rates*post_mask,input_FR) * self.dt

        # weight decay due to weight homeostasis
        decay = self.decay_r * self.W * self.dt
        decay_input_w = self.decay_r * self.input_w * self.dt

        # applying the weight change
        self.W += (hebbian_dw - decay )
        self.input_w += (hebbian_d_inputw - decay_input_w)

        # intresnsic excitability is modulated by FR with a maximum value of 2
        self.exc *= self.rates

        self.exc = torch.clamp(self.exc, 0, 2)
        self.W = torch.clamp(self.W,0,1)
        self.input_w = torch.clamp(self.input_w,0,2)
        return self.rates.detach().clone()
    
    def get_excitability(self):
        return self.exc.detach().clone()

n = 100
ni = 100
act_threshold = 1
nn = twolayermodel(ni,n,tau=50.0,dt = 1.0,lr = 0.05,threshold=0.4)
timesteps = 200
input = torch.abs(torch.normal(mean=0.0, std=1.0, size=(n,)))

FC_input = input
FC_input[:10] += 2

nonFC_input = input
nonFC_input[10:] += 1
FR_history = []
EX_history = []
rec_weights = []
ff_weights = []

rec_weights.append(nn.W.detach().clone().numpy())
ff_weights.append(nn.input_w.detach().clone().numpy())

Input_to_network = FC_input
for t in range(timesteps):
    next_FR = nn.step(Input_to_network)
    frs = (next_FR.numpy() > act_threshold)
    FR_history.append(next_FR.numpy()*frs)
    EX_history.append(nn.get_excitability().numpy())

rec_weights.append(nn.W.detach().clone().numpy())
ff_weights.append(nn.input_w.detach().clone().numpy())

input_history = np.tile(Input_to_network.cpu().numpy(), (timesteps, 1))


# single offline sessio
t_off = 200
Nrep = 10
binary_array = torch.randint(0, 2, (10,))
print(binary_array)
for i in range(Nrep):
    if binary_array[i] == 0:
        Input_to_network = FC_input
    else:
        Input_to_network = nonFC_input
    for t in range(t_off):
        next_FR = nn.step(Input_to_network)
        frs = (next_FR.numpy() > act_threshold)
        FR_history.append(next_FR.numpy()*frs)
        EX_history.append(nn.get_excitability().numpy())
    input_at_t = np.tile(Input_to_network, (t_off, 1))
    breakpoint()
    input_history = np.concatenate((input_history,input_at_t))
    # input_history = np.concatenate([input_history, input_at_t[np.newaxis, :]], axis=0)

rec_weights.append(nn.W.detach().clone().numpy())
ff_weights.append(nn.input_w.detach().clone().numpy())

FR_history = np.stack(FR_history)
EX_history = np.stack(EX_history)


plot_activity_n_excitability_time([FR_history.T,EX_history.T,input_history],
                       titles=['Neuronal Activity',
                                "Neuronal Excitability",
                                "Input FR"],
                       fname="./plots/Reimagined/Activity_n_excitability.png",
                       cmaps=['Blues', 'Greens',"Reds"])
plot_weights_over_time(rec_weights,
                       titles= ["before","after"] ,
                       fname="./plots/Reimagined/Rec_w.png",
                       cmaps='gray_r')
plot_weights_over_time(ff_weights,
                       titles= ["before","after"] ,
                       fname="./plots/Reimagined/FF_w.png",
                       cmaps='gray_r')