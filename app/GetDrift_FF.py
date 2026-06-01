import matplotlib.pyplot as plt
import json
import numpy as np
from plotting_widget import *
import torch 
import torch.nn.functional as F
from Utilities import average_firing_rates_with_active,ensamble_overlap
from tqdm import trange
start_seed = 120
torch.manual_seed(start_seed)
class twolayer_FF:
    def __init__(self, n_inp, n_neurons,n_cont,baseline_e,tau=20.0, dt=1.0, act=F.softplus, lr=1/800, decay_r=1/1000, I0=1, I1=0.05, I2=0.001):
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
        self.plas_threshold = 0
        self.act_threshold = np.abs(torch.normal(0,0.7,size=(n_neurons,)))
        self.n_cont = n_cont
        self.cont_exc = torch.abs(torch.normal(0,1,size=(self.n_cont,)))
        self.act_threshold_cont = np.abs(torch.normal(0,0.7,size=(self.n_cont,)))
        # self.threshold = threshold
        
        # Initialize random input weights
        self.input_w = torch.abs(torch.normal(0,0.05,size=(n_inp,n_neurons)))
        # self.input_w = torch.clamp(self.input_w, 0.0, 0.1)
        self.rec_w = torch.zeros(n_neurons, n_neurons)
        self.rec_w_cont = torch.zeros(self.n_cont, self.n_cont)
        
        # Zero initial rate state
        self.rates_cont = torch.zeros(self.n_cont)
        self.rates = torch.zeros(n_neurons)
        # self.
    def step(self, input_FR,cont_INP):
        """
        Perform one timestep of rate dynamics:
        input_vector: shape [n_neurons]
        """
        # calculating the input to the RNN
        input_vector = input_FR + self.rec_w @ self.rates
        cont_inp = cont_INP + self.rec_w_cont @ self.rates_cont
        # breakpoint()
        # print(input_vector.max())
        # blanket inhibition to the RNN
        I_inhib = self.I0 + self.I1 * torch.sum(self.rates) + self.I2 * torch.sum(self.rates**2)
        I_inhib_cont = self.I0 + self.I1 * torch.sum(self.rates_cont) + self.I2 * torch.sum(self.rates_cont**2)


        # print(I_inhib)
        # total input to the RNN
        input_current =  input_vector -  I_inhib 
        input_cont = cont_inp - I_inhib_cont

        # rate change as the nonlinear ODE
        dr_dt = (-self.rates +   self.act(self.excitability + input_current + self.act_threshold)) / self.tau
        dr_dt_cont = (-self.rates_cont +   self.act(self.cont_exc + input_cont + self.act_threshold_cont)) / self.tau

        
        self.rates += (dr_dt * self.dt)
        self.rates_cont += (dr_dt_cont * self.dt)

        # print(self.rates.max())
        # post_mask = (self.rates > self.threshold).float()
        post_mask = self.rates > self.plas_threshold
        # hebbian plasticity in RNN weights
        hebbian_dw = self.lr * torch.outer(self.rates*post_mask, self.rates) * self.dt
        decay = self.decay_r * self.rec_w * self.dt
        # hebbian plasticity in input weights
        self.rec_w += (hebbian_dw - decay)

        hebbian_dw_cont = self.lr * torch.outer(self.rates_cont, self.rates_cont) * self.dt
        decay_cont = self.decay_r * self.rec_w_cont * self.dt

        self.rec_w_cont += (hebbian_dw_cont - decay_cont)

        # hebbian plasticity in RNN weights
        # hebbian_dw_inp = self.lr * torch.outer(input_FR,self.rates*post_mask ) * self.dt
        # decay_inp = self.decay_r*0.1 * self.input_w * self.dt
        # # hebbian plasticity in input weights
        # self.input_w += (hebbian_dw_inp - decay_inp)

        # self.rates = torch.clamp(self.rates, 0.0, 15)  # Ensure rates are non-negativeå
        self.rec_w = torch.clamp(self.rec_w, 0.0, 1.0)  # Ensure weights are non-negative
        self.rec_w_cont = torch.clamp(self.rec_w_cont, 0.0, 1.0)
        # self.input_w = torch.clamp(self.input_w, 0.0, 0.2)  # Ensure weights are non-negative
        # self._normalize_input_outgoing(target_sum=15)
        return torch.cat((self.rates_cont.detach().clone(),self.rates.detach().clone()))
    
    def _normalize_input_outgoing(self, target_sum=None, eps=1e-12):
        """Normalize columns so that for each input feature, the outgoing weights sum to target_sum."""
        if target_sum is None:
            target_sum = self.target_out_sum
        # columns correspond to inputs if input_w is (n_neurons, n_inp)
        col_sums = self.rec_w.sum(dim=1, keepdim=True)  # shape (1, n_inp)
        # avoid division by zero: if a column is all zeros, leave it unchanged
        scale = torch.where(col_sums > eps, target_sum / col_sums, torch.ones_like(col_sums))
        self.rec_w = self.rec_w * scale  # broadcast over rows


FR_history = []
EX_history = []
rec_weights = []
ff_weights = []
last_activity = []
input_history = []
#
n = 140
n_inp = 140
n_cont = 4
E_fl = 1.8
E_fe = 1.5
E_ref = 0.7
threshold = 2
off_set = 0

# base_E[:off_set] += 2
FC_inp = 25
input = 20*torch.ones(n_inp)
cont_inp = 12*torch.ones(n_cont)
zero_cont_inp = torch.zeros(n_cont)
# input[:10] = FC_inp
off_input = input#18*torch.ones(n_inp)
# recall_input = off_input.clone()
# recall_input[:20] = FC_inp
# off_input[:off_set] -= 0
noisy_input = torch.normal(0,1,size=(n_inp,))
# input = noisy_input
zero_input = torch.zeros(n_inp)

ID = 1000

NUM_SIM = 15
N_off_days = 7
t_off = 100
IR = 100
Nrep = 10

FR_history_all = []
EX_history_all = []
rec_weights_all = []
ff_weights_all = []
last_activity_all = []
input_history_all = []

for i in trange(NUM_SIM):
    torch.manual_seed(start_seed + i)
    base_E = torch.abs(torch.normal(0,1,size=(n,)))
    nn = twolayer_FF(n_inp=n_inp, n_neurons=n,n_cont=n_cont, baseline_e = base_E.clone(), tau=20.0, dt=1, act=torch.relu, lr=1/800, decay_r=1/1000, I0=8, I1=0.7, I2=0.05)
    FR_history = []
    EX_history = []
    rec_weights = []
    ff_weights = []
    last_activity = []
    input_history = []
    rep_Activity = []
    high_threshold = 5 
    for day in range(N_off_days):
        day_activity = []
        # torch.manual_seed(120+day)
        nn.excitability = base_E.clone()
        nn.excitability[off_set+(day)*20:off_set+(day)*20+20] += E_fl
        inp_to_network = input
        if day == 0 or day == 6:
            cont_inp_to_network = cont_inp
        else:
            cont_inp_to_network = zero_cont_inp
                
        for rep in range(Nrep):
            for t in range(t_off):
                next_FR = nn.step(inp_to_network,cont_inp_to_network)
                FR_history.append(next_FR.detach().clone().numpy())
                EX_history.append(nn.excitability.detach().clone().numpy())
                input_history.append(input.numpy())
            day_activity.append(np.mean(FR_history[-t_off:],axis=0))    
            for t in range(IR):
                next_FR = nn.step(zero_input,zero_cont_inp)
                FR_history.append(next_FR.detach().clone().numpy())
                EX_history.append(nn.excitability.detach().clone().numpy())
                input_history.append(zero_input.numpy())
        rep_Activity.append(day_activity)
        rec_weights.append(nn.rec_w.detach().clone().numpy())
        # ff_weights.append(nn.input_w.detach().clone().numpy())
        last_activity.append(np.mean(day_activity,axis=0))
        for t in range(ID):
            next_FR = nn.step(zero_input,zero_cont_inp)
            FR_history.append(next_FR.detach().clone().numpy())
            EX_history.append(nn.excitability.detach().clone().numpy())
            input_history.append(zero_input.numpy())
    # breakpoint()
    FR_history_all.append(FR_history)
    EX_history_all.append(EX_history)
    rec_weights_all.append(rec_weights)
    ff_weights_all.append(ff_weights)
    last_activity_all.append(last_activity)
    input_history_all.append(input_history)

FR_history_all = np.stack(FR_history_all)
# input_history = np.stack(input_history)
EX_history_all = np.stack(EX_history_all)
last_activity_all = np.stack(last_activity_all)
np.save("./data/Reimagined_3/FR_history.npy",FR_history_all)
np.save("./data/Reimagined_3/EX_history.npy",EX_history_all)
np.save("./data/Reimagined_3/last_activity.npy",last_activity_all)
# np.save("./data/Reimagined_3/input_history.npy",input_history_all)  
sim_params = {
    "n": n,
    "n_inp": n_inp,
    "E_fl": E_fl,
    "threshold": threshold,
    "ID": ID,
    "N_off_days": N_off_days,
    "t_off": t_off,
    "IR": IR,
    "Nrep": Nrep,
    "start_seed": start_seed  # if you want reproducibility
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
filename = "./plots/Reimagined_3/all_params.json"

with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    # print(f"All parameters saved to {filename}")
plot_corr_matrix(last_activity, fname="./plots/Reimagined_3/corr_matrix.png")

FR_history_th = (FR_history_all > threshold).astype(float)*FR_history_all
breakpoint()
plot_activity_n_excitability_time([FR_history_th[0].T,EX_history_all[0].T],
                       titles=['Neuronal Activity',
                                "Neuronal Excitability"],
                       fname="./plots/Reimagined_3/Activity_n_excitability.png",
                       cmaps=['Blues', 'Greens'])
labs = ["FC"] + [f"Off {i+1}" for i in range(N_off_days)]
plot_weights_over_time(rec_weights_all[0],
                       titles=  labs,
                       fname="./plots/Reimagined_3/Rec_w.png",
                       cmaps='gray_r')

# counts = np.sum(last_activity[0,0] > threshold, axis=0)
# first_session_activity = last_activity[0,0, :]
# active_mask = counts > 0

# bins = np.arange(0, N_off_days+1,1) 
# plt.figure(figsize=(7,5))
# plt.hist(counts[1:],bins=N_off_days, density=True, align='left', rwidth=0.8, color='skyblue', edgecolor='black')
# plt.xticks(bins[1:])
# plt.xlabel("Number of sessions neuron was active")
# plt.ylabel("Number of neurons")
# plt.title("Distribution of neuron activity across sessions")
# plt.grid(True, linestyle="--", alpha=0.5)
# save_plot("./plots/Reimagined_3/session_hist.png")
# plt.show()

# FC_active_neurons = np.where(last_activity[0, :] > threshold)[0]
# print("Neurons active in FC:", FC_active_neurons)
# counts_active = counts[FC_active_neurons]
# first_session_activity_active = first_session_activity[FC_active_neurons]

# print("Remaining neurons:", counts_active.shape[0])

cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
xlabs = ["Off 1","Off 2","Off 3","Off 4","Off 5","Recall"]
Title = "Ensemble similarity of encoding and offline + recall"
# plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined_3/encoding_corr.png", use_bar_plot=True)
mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    last_activity_all,                # shape: (sims, time, neurons)
    ref_time_idx=0,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=False,
    title="Encoding vs. others (mean ± SD across sims)",
    fname="./plots/Reimagined_3/encoding_vs_others_mean_std.png",
    cmap="Reds",
    bar_plot=True
)

xlabs = ["Encoding", "Off 1","Off 2","Off 3","Off 4","Off 5"]
Title = "Ensemble similarity of recall and offline + encoding"
# plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined_3//Recall_corr.png", use_bar_plot=True)
last_activity_all, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    last_activity_all,                # shape: (sims, time, neurons)
    ref_time_idx=-1,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=False,
    title="Encoding vs. others (mean ± SD across sims)",
    fname="./plots/Reimagined_3/recall_vs_others_mean_std.png",
    cmap="Reds",
    bar_plot=True

)

