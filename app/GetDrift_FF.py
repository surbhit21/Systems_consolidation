import matplotlib.pyplot as plt
import json
import numpy as np
from plotting_widget import *
import torch 
import torch.nn.functional as F
from Utilities import average_firing_rates_with_active,ensamble_overlap
class twolayer_FF:
    def __init__(self, n_inp, n_neurons,baseline_e,tau=20.0, dt=1.0, act=torch.relu, lr=1/800, decay_r=1/1000, I0=1, I1=0.05, I2=0.001):
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
        self.plas_threshold = 1
        # self.threshold = threshold
        
        # Initialize random input weights
        self.input_w = torch.rand(n_neurons,n_inp,) 
        self.rec_w = torch.zeros(n_neurons, n_neurons)
        
        # Zero initial rate state
        self.rates = torch.zeros(n_neurons)
    
    def step(self, input_FR):
        """
        Perform one timestep of rate dynamics:
        input_vector: shape [n_neurons]
        """
        # calculating the input to the RNN
        input_vector = input_FR #@ self.input_w

        # blanket inhibition to the RNN
        I_inhib = self.I0 + self.I1 * torch.mean(self.rates) + self.I2 * torch.mean(self.rates**2)

        # total input to the RNN
        input_current =  input_vector / I_inhib 

        # rate change as the nonlinear ODE
        dr_dt = (-self.rates +   self.act(self.excitability + input_current)) / self.tau

        self.rates += (dr_dt * self.dt)
        # print(self.rates.max())
        # post_mask = (self.rates > self.threshold).float()
        post_mask = self.rates > self.plas_threshold
        # hebbian plasticity in RNN weights
        hebbian_dw = self.lr * torch.outer(self.rates*post_mask, self.rates) * self.dt
        decay = self.decay_r * self.rec_w * self.dt
        # hebbian plasticity in input weights
        self.rec_w += (hebbian_dw - decay)

        # hebbian plasticity in RNN weights
        # hebbian_dw = self.lr * torch.outer(self.rates, input_FR) * self.dt
        # decay = 0#self.decay_r * self.rec_w * self.dt
        # hebbian plasticity in input weights
        # self.input_w += (hebbian_dw - decay)

        self.rec_w = torch.clamp(self.rec_w, 0.0, 1.0)  # Ensure weights are non-negative
        # self.input_w = torch.clamp(self.input_w, 0.0, 1.0)  # Ensure weights are non-negative
        return self.rates.detach().clone()

torch.manual_seed(2025)
FR_history = []
EX_history = []
rec_weights = []
ff_weights = []
last_activity = []
input_history = []
#
n = 140
n_inp = 140
E_fl = 1.5
E_ref = 0
threshold = 0
base_E = torch.abs(torch.normal(0,0.5,size=(n,)))
nn = twolayer_FF(n_inp=n_inp, n_neurons=n, baseline_e = base_E.clone(), tau=20.0, dt=0.1, act=torch.relu, lr=1/800, decay_r=1/1000, I0=1, I1=0.3, I2=0)
input = 10*torch.ones(n_inp)
noisy_input = torch.normal(0,1,size=(n_inp,))
# input = noisy_input
zero_input = torch.zeros(n_inp)
t_FC = 500
ID = 1000
nn.excitability[:20] += E_fl
# rec_weights.append(nn.rec_w.detach().clone().numpy())
for t in range(t_FC):
    next_FR = nn.step(input)
    FR_history.append(next_FR.detach().clone().numpy())
    EX_history.append(nn.excitability.detach().clone().numpy())
    input_history.append(input.numpy)
rec_weights.append(nn.rec_w.detach().clone().numpy())
# ff_weights.append(nn.input_w.detach().clone().numpy())
last_activity.append(np.mean(FR_history[-100:],axis=0))  
for t in range(ID):
        next_FR = nn.step(zero_input)
        FR_history.append(next_FR.detach().clone().numpy())
        EX_history.append(nn.excitability.detach().clone().numpy())
        input_history.append(zero_input.numpy())
frs = np.array(FR_history)
mean_FR = frs.mean(axis=1)
# breakpoint()
plt.plot(np.arange(0,frs.shape[0],1)*0.001,mean_FR,label = "E")
plt.xlabel("Time (s)")
plt.ylabel("Mean FR (Hz)")
plt.title("Mean Firing Rate during FC")
plt.legend()
plt.savefig("./plots/Reimagined/Mean_FR_FC.png")
plt.show()
plt.close()


N_off_days = 5
t_off = 200
IR = 100
Nrep = 3
for day in range(N_off_days):
    nn.excitability = base_E.clone()
    # nn.excitability[day*20:(day)*20+20] -= (E_ref)
    nn.excitability[(day+1)*20:(day+1)*20+20] += E_fl
    for rep in range(Nrep):
        for t in range(t_off):
            next_FR = nn.step(input*0.8)
            FR_history.append(next_FR.detach().clone().numpy())
            EX_history.append(nn.excitability.detach().clone().numpy())
            input_history.append(input.numpy())
            
        for t in range(IR):
            next_FR = nn.step(zero_input)
            FR_history.append(next_FR.detach().clone().numpy())
            EX_history.append(nn.excitability.detach().clone().numpy())
            input_history.append(zero_input.numpy())
    rec_weights.append(nn.rec_w.detach().clone().numpy())
    # ff_weights.append(nn.input_w.detach().clone().numpy())
    last_activity.append(np.mean(FR_history[-300:-100],axis=0))
    for t in range(ID):
        next_FR = nn.step(zero_input)
        FR_history.append(next_FR.detach().clone().numpy())
        EX_history.append(nn.excitability.detach().clone().numpy())
        input_history.append(zero_input.numpy())

  
t_recall = 500
ID = 1000
nn.excitability = base_E.clone()
# nn.excitability[(N_off_days)*20:(N_off_days)*20+20] -= (E_ref)
nn.excitability[-20:] += E_fl
for t in range(t_FC):
    next_FR = nn.step(input)
    FR_history.append(next_FR.detach().clone().numpy())
    EX_history.append(nn.excitability.detach().clone().numpy())
    input_history.append(input.numpy)
rec_weights.append(nn.rec_w.detach().clone().numpy())
# ff_weights.append(nn.input_w.detach().clone().numpy())
last_activity.append(np.mean(FR_history[-100:],axis=0))  
for t in range(ID):
        next_FR = nn.step(zero_input)
        FR_history.append(next_FR.detach().clone().numpy())
        EX_history.append(nn.excitability.detach().clone().numpy())
        input_history.append(zero_input.numpy())


breakpoint()



FR_history = np.stack(FR_history)
# input_history = np.stack(input_history)
EX_history = np.stack(EX_history)
last_activity = np.stack(last_activity)
sim_params = {
    "n": n,
    "n_inp": n_inp,
    "E_fl": E_fl,
    "threshold": threshold,
    "t_FC": t_FC,
    "ID": ID,
    "N_off_days": N_off_days,
    "t_off": t_off,
    "IR": IR,
    "Nrep": Nrep,
    "t_recall": t_recall,
    "seed": 2025  # if you want reproducibility
}
data = {
        "model_params": {
            "n_neurons": nn.n_neurons,
            "tau": nn.tau,
            "dt": nn.dt,
            "lr": nn.lr,
            "decay_r": nn.decay_r,
            "I0": nn.I0,
            "I1": nn.I1,
            "I2": nn.I2,
            # "excitability": model.excitability.detach().cpu().numpy().tolist(),
        },
        "simulation_params": sim_params
    }
filename = "./plots/Reimagined/all_params.json"
with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    # print(f"All parameters saved to {filename}")
plot_corr_matrix(last_activity, fname="./plots/Reimagined/corr_matrix.png")

avg_FC, active_FC, avg_days, active_days, avg_recall, active_recall \
    = average_firing_rates_with_active(FR_history.T, 
                                    T_FC=t_FC,
                                    T_offline=t_off, 
                                    T_ir=IR,
                                    Nday=N_off_days, 
                                    Nrep=Nrep, 
                                    T_recall=t_recall, 
                                    ID=ID,
                                    threshold=threshold)

en_recall_overlap = ensamble_overlap(active_FC, [active_recall])
en_off_overlap = ensamble_overlap(active_FC, active_days)
re_off_overlap = ensamble_overlap(active_recall, active_days)
print("Ensemble overlap between encoding and recall: \n", len(en_recall_overlap[0])/len(active_FC))
print("Ensemble overlap between encoding and offline: \n", [len(x)/len(active_FC) for x in en_off_overlap])
print("Ensemble overlap between recall and offline: \n", [len(x)/ len(active_recall) for x in re_off_overlap])
FR_history_th = (FR_history > threshold).astype(float)*FR_history
plot_activity_n_excitability_time([FR_history_th.T,EX_history.T],
                       titles=['Neuronal Activity',
                                "Neuronal Excitability"],
                       fname="./plots/Reimagined/Activity_n_excitability.png",
                       cmaps=['Blues', 'Greens'])
labs = ["FC"] + [f"Off {i+1}" for i in range(N_off_days)]
plot_weights_over_time(rec_weights,
                       titles=  labs,
                       fname="./plots/Reimagined/Rec_w.png",
                       cmaps='gray_r')
# plot_weights_over_time(ff_weights,
#                        titles= ["FC","Of1","Of2","Of3","Of4","Of5","Recall"] ,
#                        fname="./plots/Reimagined/FF_w.png",
#                        cmaps='gray_r',plot_title="CA3 -> CA1 weights over time")


breakpoint()
xlabs = ["Off 1","Off 2","Off 3","Off 4","Off 5","Recall"]
Title = "Ensemble similarity of encoding and offline + recall"
plot_row_correlations(avg_FC.T,np.column_stack([avg_days.mean(axis=2),avg_recall]).T, xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr.png", use_bar_plot=True)

xlabs = ["Encoding", "Off 1","Off 2","Off 3","Off 4","Off 5"]
Title = "Ensemble similarity of recall and offline + encoding"
plot_row_correlations(avg_recall.T,np.column_stack([avg_FC, avg_days.mean(axis=2)]).T, xlabs=xlabs,title=Title,fname="./plots/Reimagined//Recall_corr.png", use_bar_plot=True)
