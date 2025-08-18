import matplotlib.pyplot as plt
import numpy as np
import os
from plotting_widget import *
import torch 
import torch.nn.functional as F
from Utilities import average_firing_rates_with_active,ensamble_overlap
class twolayer_FF:
    def __init__(self, n_inp, n_neurons,baseline_e,tau=10.0, dt=1.0, act=torch.relu, lr=8e-2, decay_r=1e-2, threshold=0, I0=1, I1=0.05, I2=0.001):
        self.n_neurons = n_neurons
        self.tau = tau  # rate constant 
        self.dt = dt  # discretization step
        # self.exc = torch.abs(torch.rand(n_neurons))  # excitability 
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
        input_current =  input_vector + self.excitability - I_inhib 

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

torch.manual_seed(2025)
n = 140
e = 3.5
n_drifti ,ni_FC = 40,10 #(40 drify and 10 contextual neurons)
ni = n_drifti + ni_FC
act_threshold = 1
base_e = torch.abs(torch.normal(mean=0.0, std=1.0, size=(n,))) * 0.01
nn = twolayer_FF(ni,n,baseline_e=base_e,tau=50.0,dt = 0.1,lr = 0.05,threshold=0)

t_FC = 200
input = torch.abs(torch.normal(mean=0.0, std=1.0, size=(ni,)))
FC_input = input.clone()
FC_input[n_drifti:] += 10
nn.excitability[:20] += e
nonFC_input = input
nonFC_input[:n_drifti] += 0
# nonFC_input[10:] += 1
FR_history = []
EX_history = []
rec_weights = []
ff_weights = []
last_activity = []
# rec_weights.append(nn.W.detach().clone().numpy())
# ff_weights.append(nn.input_w.detach().clone().numpy())

Input_to_network = FC_input
for t in range(t_FC):
    next_FR = nn.step(Input_to_network)
    # frs = (next_FR.numpy() > act_threshold)
    FR_history.append(next_FR.numpy())
    EX_history.append(nn.excitability.detach().clone().numpy())

# rec_weights.append(nn.W.detach().clone().numpy())
ff_weights.append(nn.input_w.detach().clone().numpy())
last_activity.append(FR_history[-1])
input_history = np.tile(Input_to_network.cpu().numpy(), (t_FC, 1))

# single offline sessio
t_off = 200
Nrep = 1
binary_array = torch.randint(0, 2, (10,))
print(binary_array)
n_off_days = 5

for day in range(n_off_days):
    torch.manual_seed(day)  # Set seed for reproducibility
    nonFC_input = torch.abs(torch.normal(mean=0.0, std=1.0, size=(ni,))) 
    nn.excitability[(day)*20:(day+1)*20] -= e
    nn.excitability[(day+1)*20:(day+2)*20] += e
    for i in range(Nrep):
        if binary_array[i] == 2:
            Input_to_network = FC_input
        else:
            Input_to_network = nonFC_input
        for t in range(t_off):
            next_FR = nn.step(nonFC_input)
            # frs = (next_FR.numpy() > act_threshold)
            FR_history.append(next_FR.numpy())
            EX_history.append(nn.excitability.detach().clone().numpy())
        input_at_t = np.tile(Input_to_network, (t_off, 1))
        # breakpoint()
        input_history = np.concatenate((input_history,input_at_t))
        # input_history = np.concatenate([input_history, input_at_t[np.newaxis, :]], axis=0)
    last_activity.append(FR_history[-1])
    ff_weights.append(nn.input_w.detach().clone().numpy())

# rec_weights.append(nn.W.detach().clone().numpy())
t_recall = 200
nn.excitability[(n_off_days)*20:(n_off_days+1)*20] -= e
nn.excitability[-20:] += e
for t in range(t_recall):
    # print((n_off_days)*20,(n_off_days+1)*20)
    next_FR = nn.step(FC_input)
    # frs = (next_FR.numpy() > act_threshold)
    FR_history.append(next_FR.numpy())
    EX_history.append(nn.excitability.detach().clone().numpy())
last_activity.append(FR_history[-1])    
ff_weights.append(nn.input_w.detach().clone().numpy())
FR_history = np.stack(FR_history)
EX_history = np.stack(EX_history)
last_activity = np.stack(last_activity)
plot_corr_matrix(last_activity, fname="./plots/Drift_FF/corr_matrix.png")

avg_FC, active_FC, avg_days, active_days, avg_recall, active_recall \
    = average_firing_rates_with_active(FR_history.T, 
                                    T_FC=t_FC,
                                    T_offline=t_off, 
                                    Nday=n_off_days, 
                                    Nrep=Nrep, 
                                    T_recall=t_recall, 
                                    threshold=act_threshold)

en_recall_overlap = ensamble_overlap(active_FC, [active_recall])
en_off_overlap = ensamble_overlap(active_FC, active_days)
re_off_overlap = ensamble_overlap(active_recall, active_days)
print("Ensemble overlap between encoding and recall: \n", en_recall_overlap/len(active_FC))
print("Ensemble overlap between encoding and offline: \n", en_off_overlap/len(active_FC))
print("Ensemble overlap between recall and offline: \n", re_off_overlap/ len(active_recall))

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
                       titles= ["FC","Of1","Of2","Of3","Of4","Of5","Recall"] ,
                       fname="./plots/Reimagined/FF_w.png",
                       cmaps='gray_r',plot_title="CA3 -> CA1 weights over time")


breakpoint()
xlabs = ["Off 1","Off 2","Off 3","Off 4","Off 5","Recall"]
Title = "Ensemble similarity of encoding and offline + recall"
plot_row_correlations(avg_FC.T,np.column_stack([avg_days,avg_recall]).T, xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr.png", use_bar_plot=True)

xlabs = ["Encoding", "Off 1","Off 2","Off 3","Off 4","Off 5"]
Title = "Ensemble similarity of recall and offline + encoding"
plot_row_correlations(avg_recall.T,np.column_stack([avg_FC, avg_days]).T, xlabs=xlabs,title=Title,fname="./plots/Reimagined//Recall_corr.png", use_bar_plot=True)
