import matplotlib.pyplot as plt
import numpy as np
import os
from plotting_widget import *
import torch 
import torch.nn.functional as F
from Utilities import average_firing_rates_with_active,ensamble_overlap
class twolayer_FF:
    def __init__(self, n_inp, nCA1,nCA3,eca1,eca3,tau=10.0, dt=1.0, act=torch.relu, lr=8e-2, decay_r=1e-2, threshold=0, I0=5, I1=0.05, I2=0.01):
        self.nCA1 = nCA1
        self.tau = tau  # rate constant 
        self.dt = dt  # discretization step
        self.act = act  # activation function
        self.lr = lr  # learning rate for synaptic weights
        self.decay_r = decay_r  # decay rate for synaptic weights
        self.I0 = I0
        self.I1 = I1
        self.I2 = I2
        self.excitability = eca1
        self.excitability_CA3 = eca3
        self.threshold = threshold
        
        # Initialize random input weights
        self.W_inp_CA1 = torch.abs(torch.randn(n_inp, nCA1)) / n_inp**0.5
        self.W_inp_CA3 = torch.abs(torch.randn(n_inp, nCA3)) / n_inp**0.5
        self.W_CA3_CA3 = torch.abs(torch.zeros(nCA3, nCA3)) / nCA3**0.5
        self.W_CA3_CA1 = torch.abs(torch.zeros(nCA3, nCA1)) / nCA3**0.5
        
        
        # Zero initial rate state
        self.r_CA1 = torch.zeros(nCA1)
        self.r_CA3 = torch.zeros(nCA3)
    
    def step(self, input_FR):
        """
        Perform one timestep of rate dynamics:
        input_vector: shape [n_inp]
        """
        # calculating the input to the CA1
        tot_inp_CA1 = input_FR @ self.W_inp_CA1 + self.W_CA3_CA1 @ self.r_CA3 

        # calculating the input to the CA3
        total_inp_CA3 = input_FR @ self.W_inp_CA3 + self.W_CA3_CA3 @ self.r_CA3
        
        # blanket inhibition to the RNN
        I_inhib = self.I0 + self.I1 * torch.sum(self.r_CA1) + self.I2 + torch.sum(self.r_CA1**2)


        # rate change as the nonlinear ODE
        dr_ca1_dt = (-self.r_CA1 +  self.act(tot_inp_CA1  - I_inhib + self.excitability)) / self.tau
        dr_ca3_dt = (-self.r_CA1 +  self.act(total_inp_CA3  - I_inhib + self.excitability_CA3 )) / self.tau

        
        self.r_CA1 += (dr_ca1_dt * self.dt)
        self.r_CA3 += (dr_ca3_dt * self.dt)
    
        post_mask_ca1 = (self.r_CA1 > self.threshold).float()
        post_mask_ca3 = (self.r_CA3 > self.threshold).float()
        
        # hebbian plasticity in inp -> CA1 weights
        hebbian_dw_ca1 = self.lr * torch.outer(input_FR, self.r_CA1*post_mask_ca1) * self.dt
        # hebbian plasticity in inp -> CA3 weights
        hebbian_dw_ca3 = self.lr * torch.outer(input_FR, self.r_CA3*post_mask_ca3) * self.dt
        # hebbian plasticity in CA3 -> CA1 weights
        hebbian_dw_ca3_ca1 = self.lr * torch.outer(self.r_CA3, self.r_CA1*post_mask_ca1) * self.dt
        # hebbian plasticity in CA3 -> CA3 weights
        hebbian_dw_ca3_ca3 = self.lr * torch.outer(self.r_CA3, self.r_CA3*post_mask_ca3) * self.dt
        
        decay_ca1 = self.decay_r * self.W_inp_CA1 * self.dt
        decay_ca3 = self.decay_r * self.W_inp_CA3 * self.dt
        decay_ca3_r = self.decay_r * self.W_CA3_CA3 * self.dt
        decay_ca3_ca1 = self.decay_r * self.W_CA3_CA1 * self.dt
        
        # hebbian plasticity in input weights
        # self.W_inp_CA1 += hebbian_dw_ca1 - decay_ca1
        # self.W_inp_CA3 += hebbian_dw_ca3 - decay_ca3
        self.W_CA3_CA3 += hebbian_dw_ca3_ca3 - decay_ca3_r
        self.W_CA3_CA1 += hebbian_dw_ca3_ca1 - decay_ca3_ca1

        self.W_inp_CA1 = torch.clamp(self.W_inp_CA1, 0.0, 1.0)  # Ensure weights are non-negative
        self.W_inp_CA3 = torch.clamp(self.W_inp_CA3, 0.0, 1.0)  # Ensure weights are non-negative
        self.W_CA3_CA3 = torch.clamp(self.W_CA3_CA3, 0.0, 1.0)  # Ensure weights are non-negative
        self.W_CA3_CA1 = torch.clamp(self.W_CA3_CA1, 0.0, 1.0)  # Ensure weights are non-negative
        return self.r_CA1.detach().clone(),self.r_CA3.detach().clone()

torch.manual_seed(2025)
n = 140
nca3 = 140
e = 3.5
e_CA3 = 2
n_drifti ,ni_FC = 40,10 #(40 drify and 10 contextual neurons)
ni = n_drifti + ni_FC
act_threshold = 1
base_e = torch.abs(torch.normal(mean=0.0, std=1.0, size=(n+nca3,))) * 0.01
nn = twolayer_FF(ni,n,nca3,base_e[:n],base_e[n:],tau=50.0,dt = 0.1,lr = 0.05,threshold=0)
t_ID = 500
t_IR = 100
t_FC = 5000
input = torch.abs(torch.normal(mean=0.0, std=1.0, size=(ni,)))
FC_input = input.clone()
FC_input[n_drifti:] += 2
nn.excitability[:20] += e
nn.excitability_CA3[:20] += e_CA3
nonFC_input = input
nonFC_input[:n_drifti] += 0
# nonFC_input[10:] += 1
CA1_FR_history = []
CA3_FR_history = []
EX_history_CA1 = []
EX_history_CA3 = []
rec_weights = []
ff_weights = []
last_activity = []
# rec_weights.append(nn.W.detach().clone().numpy())
# ff_weights.append(nn.W_CA3_CA1.detach().clone().numpy())

Input_to_network = FC_input
for t in range(t_FC):
    next_FR_CA1,next_FR_CA3 = nn.step(Input_to_network)
    # frs = (next_FR_CA1.numpy() > act_threshold)
    CA1_FR_history.append(next_FR_CA1.numpy())
    CA3_FR_history.append(next_FR_CA3.numpy())
    EX_history_CA1.append(nn.excitability.detach().clone().numpy())
    EX_history_CA3.append(nn.excitability_CA3.detach().clone().numpy())

# rec_weights.append(nn.W.detach().clone().numpy())
ff_weights.append(nn.W_CA3_CA1.detach().clone().numpy())
rec_weights.append(nn.W_CA3_CA3.detach().clone().numpy())
last_activity.append(CA1_FR_history[-1])
input_history = np.tile(Input_to_network.cpu().numpy(), (t_FC, 1))

# single offline sessio
t_off = 200
Nrep = 10
binary_array = torch.randint(0, 2, (10,))
print(binary_array)
n_off_days = 5

for day in range(n_off_days):
    torch.manual_seed(day)  # Set seed for reproducibility
    nonFC_input = torch.abs(torch.normal(mean=0.0, std=1.0, size=(ni,))) 
    nn.excitability[(day)*20:(day+1)*20] -= e
    nn.excitability[(day+1)*20:(day+2)*20] += e
    nn.excitability_CA3[(day)*20:(day+1)*20] -= e_CA3
    nn.excitability_CA3[(day+1)*20:(day+2)*20] += e_CA3
    for i in range(Nrep):
        if binary_array[i] == 2:
            Input_to_network = FC_input
        else:
            Input_to_network = nonFC_input
        for t in range(t_off):
            next_FR_CA1,next_FR_CA3 = nn.step(FC_input)
            # frs = (next_FR_CA1.numpy() > act_threshold)
            CA1_FR_history.append(next_FR_CA1.numpy())
            CA3_FR_history.append(next_FR_CA3.numpy())
            EX_history_CA1.append(nn.excitability.detach().clone().numpy())
            EX_history_CA3.append(nn.excitability_CA3.detach().clone().numpy())
        input_at_t = np.tile(Input_to_network, (t_off, 1))
        # breakpoint()
        input_history = np.concatenate((input_history,input_at_t))
        # input_history = np.concatenate([input_history, input_at_t[np.newaxis, :]], axis=0)
    last_activity.append(CA1_FR_history[-1])
    ff_weights.append(nn.W_CA3_CA1.detach().clone().numpy())
    rec_weights.append(nn.W_CA3_CA3.detach().clone().numpy())
# rec_weights.append(nn.W.detach().clone().numpy())
t_recall = 200
nn.excitability[(n_off_days)*20:(n_off_days+1)*20] -= e
nn.excitability[-20:] += e
nn.excitability_CA3[(n_off_days)*20:(n_off_days+1)*20] -= e_CA3
nn.excitability_CA3[-20:] += e_CA3
Input_to_network = FC_input
input_at_t = np.tile(Input_to_network, (t_recall, 1))
input_history = np.concatenate((input_history,input_at_t))
for t in range(t_recall):
    # print((n_off_days)*20,(n_off_days+1)*20)
    next_FR_CA1,next_FR_CA3 = nn.step(Input_to_network)
    # frs = (next_FR_CA1.numpy() > act_threshold)
    CA1_FR_history.append(next_FR_CA1.numpy())
    CA3_FR_history.append(next_FR_CA3.numpy())
    EX_history_CA1.append(nn.excitability.detach().clone().numpy())
    EX_history_CA3.append(nn.excitability_CA3.detach().clone().numpy())
last_activity.append(CA1_FR_history[-1])    
ff_weights.append(nn.W_CA3_CA1.detach().clone().numpy())
rec_weights.append(nn.W_CA3_CA3.detach().clone().numpy())
CA1_FR_history = np.stack(CA1_FR_history)
CA3_FR_history = np.stack(CA3_FR_history)
EX_history_CA1 = np.stack(EX_history_CA1)
EX_history_CA3 = np.stack(EX_history_CA3)
last_activity = np.stack(last_activity)
plot_corr_matrix(last_activity, fname="./plots/Drift_FF/corr_matrix.png")

avg_FC, active_FC, avg_days, active_days, avg_recall, active_recall \
    = average_firing_rates_with_active(CA1_FR_history.T, 
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

plot_activity_n_excitability_time([CA1_FR_history.T,CA3_FR_history.T,EX_history_CA1.T,EX_history_CA3.T,input_history.T],
                       titles=['Neuronal Activity (CA1)',
                               'Neuronal Activity (CA3)',
                                "Neuronal Excitability (CA1)",
                                "Neuronal Excitability (CA3)",
                                "Input FR"],
                       fname="./plots/Reimagined/Activity_n_excitability.png",
                       cmaps=['Blues','Blues', 'Greens','Greens',"Reds"])
# plot_weights_over_time(rec_weights,
#                        titles= ["before","after"] ,
#                        fname="./plots/Reimagined/Rec_w.png",
#                        cmaps='gray_r')
plot_weights_over_time(ff_weights,
                       titles= ["FC","Of1","Of2","Of3","Of4","Of5","Recall"] ,
                       fname="./plots/Reimagined/FF_w.png",
                       cmaps='gray_r',plot_title="CA3 -> CA1 weights over time")
plot_weights_over_time(rec_weights,
                       titles= ["FC","Of1","Of2","Of3","Of4","Of5","Recall"] ,
                       fname="./plots/Reimagined/Rec_w.png",
                       cmaps='gray_r',plot_title="CA3 -> CA3 weights over time")


breakpoint()
xlabs = ["Off 1","Off 2","Off 3","Off 4","Off 5","Recall"]
Title = "Ensemble similarity of encoding and offline + recall"
plot_row_correlations(avg_FC.T,np.column_stack([avg_days,avg_recall]).T, xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr.png", use_bar_plot=True)

xlabs = ["Encoding", "Off 1","Off 2","Off 3","Off 4","Off 5"]
Title = "Ensemble similarity of recall and offline + encoding"
plot_row_correlations(avg_recall.T,np.column_stack([avg_FC, avg_days]).T, xlabs=xlabs,title=Title,fname="./plots/Reimagined/Recall_corr.png", use_bar_plot=True)
