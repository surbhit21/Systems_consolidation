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
    def __init__(self, n_inp, n_MTL,n_CTX,baseline_e,base_e_ctx,tau=20.0, dt=1.0, act=F.softplus, 
                 lr=1/800, decay_r=1/1000, 
                 lr_ctx = 1/1200, decay_r_ctx = 1e-5,
                 I0=1, I1=0.05, I2=0.001):
        self.n_MTL = n_MTL
        self.n_CTX = n_CTX
        self.tau = tau  # rate constant 
        self.dt = dt  # discretization step
        # self.exc = torch.abs(torch.rand(n_MTL))  # excitability 
        self.act = act  # activation function
        self.lr = lr  # learning rate for synaptic weights
        self.decay_r = decay_r  # decay rate for synaptic weights
        self.lr_ctx = lr_ctx  # learning rate for synaptic weights
        self.decay_r_ctx = decay_r_ctx
        self.I0 = I0
        self.I1 = I1
        self.I2 = I2
        self.excitability = baseline_e
        self.excitability_ctx = base_e_ctx
        self.plas_threshold = 0
        self.act_threshold = np.abs(torch.normal(0,0.7,size=(n_MTL,)))
        self.threshold = threshold
        self.FF_MTL_CTX = 0
        self.FB_CTX_MTL = 0
        self.op_neuron = 1
        
        # Initialize random input weights
        self.input_w = torch.abs(torch.normal(0,0.05,size=(n_inp,n_MTL)))
        # self.input_w = torch.clamp(self.input_w, 0.0, 0.1)
        self.rec_w = torch.zeros(n_MTL, n_MTL)
        self.rec_w_ctx = torch.zeros(self.n_CTX, self.n_CTX)
        
        self.mtl_op_w = torch.zeros(self.op_neuron,n_MTL)
        self.ctx_op_w = torch.zeros(self.op_neuron,n_MTL)
        # Zero initial rate state
        self.rates_ctx = torch.zeros(self.n_CTX)
        self.rates = torch.zeros(n_MTL)
        # self.
    def TurnOFF_FB(self):
        self.FB_CTX_MTL = 0
    def TurnON_FB(self):
        self.FB_CTX_MTL = 1
    def TurnOFF_FF(self):
        self.FF_MTL_CTX = 0
    def TurnON_FF(self):
        self.FF_MTL_CTX = 0
    def step(self, input_FR):
        """
        Perform one timestep of rate dynamics:
        input_vector: shape [n_MTL]
        """
        # calculating the input to the RNN
        input_vector = input_FR + self.rec_w @ self.rates - self.FB_CTX_MTL * self.rates_ctx
        input_CTX = input_FR + self.rec_w_ctx @ self.rates_ctx + self.FF_MTL_CTX * self.rates
        # breakpoint()
        # print(input_vector.max(),input_CTX.max(),self.rec_w.max(),self.rec_w_ctx.max())
        # blanket inhibition to the RNN
        I_inhib = self.I0 + self.I1 * torch.sum(self.rates) + self.I2 * torch.sum(self.rates**2)
        I_inhib_ctx = self.I0 + self.I1 * torch.sum(self.rates_ctx) + self.I2 * torch.sum(self.rates_ctx**2)


        # print(I_inhib)
        # total input to the RNN
        input_current =  input_vector -  I_inhib 
        input_ctx = input_CTX - I_inhib_ctx

        # rate change as the nonlinear ODE
        dr_dt = (-self.rates +   self.act(self.excitability + input_current + self.act_threshold)) / self.tau
        dr_dt_ctx = (-self.rates_ctx +   self.act(  input_ctx )) / self.tau

        
        self.rates += (dr_dt * self.dt)
        self.rates_ctx += (dr_dt_ctx * self.dt)

        # print(self.rates.max(),self.rates_ctx.max())
        # post_mask = (self.rates > self.threshold).float()
        post_mask = self.rates > self.plas_threshold
        # hebbian plasticity in RNN weights
        hebbian_dw = self.lr * torch.outer(self.rates*post_mask, self.rates*post_mask) * self.dt
        decay = self.decay_r * self.rec_w * self.dt

        # hebbian plasticity in input weights
        self.rec_w += (hebbian_dw - decay)

        post_mask = self.rates_ctx > self.plas_threshold
        hebbian_dw_ctx = self.lr_ctx * torch.outer(self.rates_ctx*post_mask, self.rates_ctx*post_mask) * self.dt
        decay_ctx = self.decay_r_ctx * self.rec_w_ctx * self.dt

        self.rec_w_ctx += (hebbian_dw_ctx - decay_ctx)

        # hebbian plasticity in RNN weights
        # hebbian_dw_inp = self.lr * torch.outer(input_FR,self.rates*post_mask ) * self.dt
        # decay_inp = self.decay_r*0.1 * self.input_w * self.dt
        # # hebbian plasticity in input weights
        # self.input_w += (hebbian_dw_inp - decay_inp)

        # self.rates = torch.clamp(self.rates, 0.0, 15)  # Ensure rates are non-negativeå
        self.rec_w = torch.clamp(self.rec_w, 0.0, 1.0)  # Ensure weights are non-negative
        self.rec_w_ctx = torch.clamp(self.rec_w_ctx, 0.0, 1.0)
        # self.input_w = torch.clamp(self.input_w, 0.0, 0.2)  # Ensure weights are non-negative
        # self._normalize_input_outgoing(target_sum=15)
        return self.rates.detach().clone(),self.rates_ctx.detach().clone()
    
    def _normalize_input_outgoing(self, target_sum=None, eps=1e-12):
        """Normalize columns so that for each input feature, the outgoing weights sum to target_sum."""
        if target_sum is None:
            target_sum = self.target_out_sum
        # columns correspond to inputs if input_w is (n_MTL, n_inp)
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
n = 100
n_inp = 100
n_ctx = 100
E_fl = 2.5
E_fe = 0
E_ref = 0.7
threshold = 5
off_set = 0

# base_E[:off_set] += 2
FC_inp = 25
input = 27*torch.ones(n_inp)
zero_input = torch.zeros(n_inp)

ID = 1000

NUM_SIM = 100
N_off_days = 5
t_off = 100
IR = 100
Nrep = 10

FR_history_all = []
EX_history_all = []
EX_history_ctx_all = []
rec_weights_all = []
ff_weights_all = []
last_activity_all = []
input_history_all = []
rec_ctx_weights_all = []
FR_ctx_history_all = []
last_activity_ctx_all = []
for i in trange(NUM_SIM):
    torch.manual_seed(start_seed + i)
    base_E = torch.abs(torch.normal(0,1,size=(n_ctx,)))
    base_e_ctx = torch.abs(torch.normal(0,1,size=(n,)))
    nn = twolayer_FF(n_inp=n_inp, n_MTL=n,n_CTX=n_ctx, baseline_e = base_E.clone(),
                     base_e_ctx=base_e_ctx.clone(), tau=20.0, dt=1, act=torch.relu,
                    lr=1/800, decay_r=1/1000, 
                    lr_ctx = 1e-4,decay_r_ctx=1e-8,
                    I0=8, I1=0.6, I2=0.05)
    FR_history = []
    FR_history_ctx = []
    EX_history = []
    EX_history_ctx = []
    rec_weights = []
    rec_ctx_weights = []
    ff_weights = []
    last_activity = []
    input_history = []
    rep_Activity = []
    last_activity_ctx = []
    high_threshold = 5 
    for day in range(N_off_days):
        day_activity = []
        day_activity_ctx = []
        # torch.manual_seed(120+day)
        nn.excitability = base_E.clone()
        nn.excitability[off_set+(day)*20:off_set+(day)*20+20] += E_fl
        inp_to_network = input                
        # if day == 1:
        #     nn.TurnOFF_FB()
        #     nn.TurnOFF_FF()
        # else:
        #     nn.TurnON_FB()
        #     nn.TurnON_FF()
        for rep in range(Nrep):
            nn.TurnON_FB()
            nn.TurnON_FF()
            for t in range(t_off):
                next_FR,FR_ctx = nn.step(inp_to_network)
                FR_history.append(next_FR)
                FR_history_ctx.append(FR_ctx)
                EX_history.append(nn.excitability.detach().clone().numpy())
                EX_history_ctx.append(nn.excitability_ctx.detach().clone().numpy())
                input_history.append(input.numpy())
            day_activity.append(np.mean(FR_history[-t_off:],axis=0))
            day_activity_ctx.append(np.mean(FR_history_ctx[-t_off:],axis=0))    
            nn.TurnOFF_FB()
            nn.TurnOFF_FF()
            for t in range(IR):
                next_FR,FR_ctx = nn.step(zero_input)
                FR_history.append(next_FR)
                FR_history_ctx.append(FR_ctx)
                EX_history.append(nn.excitability.detach().clone().numpy())
                EX_history_ctx.append(nn.excitability_ctx.detach().clone().numpy())
                input_history.append(zero_input.numpy())
        rep_Activity.append(day_activity)
        rec_weights.append(nn.rec_w.detach().clone().numpy())
        rec_ctx_weights.append(nn.rec_w_ctx.detach().clone().numpy())
        # ff_weights.append(nn.input_w.detach().clone().numpy())
        last_activity.append(np.mean(day_activity,axis=0))
        last_activity_ctx.append(np.mean(day_activity_ctx,axis=0))
        for t in range(ID):
            next_FR,FR_ctx = nn.step(zero_input)
            FR_history.append(next_FR)
            FR_history_ctx.append(FR_ctx)
            EX_history.append(nn.excitability.detach().clone().numpy())
            EX_history_ctx.append(nn.excitability_ctx.detach().clone().numpy())
            input_history.append(zero_input.numpy())
    # breakpoint()
    FR_history_all.append(FR_history)
    FR_ctx_history_all.append(FR_history_ctx)
    EX_history_all.append(EX_history)
    EX_history_ctx_all.append(EX_history_ctx)
    rec_weights_all.append(rec_weights)
    rec_ctx_weights_all.append(rec_ctx_weights)
    ff_weights_all.append(ff_weights)
    last_activity_all.append(last_activity)
    input_history_all.append(input_history)
    last_activity_ctx_all.append(last_activity_ctx)

FR_history_all = np.stack(FR_history_all)
FR_ctx_history_all = np.stack(FR_ctx_history_all)
# input_history = np.stack(input_history)
EX_history_all = np.stack(EX_history_all)
EX_history_ctx_all = np.stack(EX_history_ctx_all)
last_activity_all = np.stack(last_activity_all)
last_activity_ctx_all = np.stack(last_activity_ctx_all)
rec_weights_all = np.stack(rec_weights_all)
rec_ctx_weights_all = np.stack(rec_ctx_weights_all)
op_data_folder = "./data/Reimagined5"
op_plot_folder = "./plots/Reimagined5"# --- IGNORE ---
os.makedirs(op_data_folder, exist_ok=True)
np.save("{}/rec_weights.npy".format(op_data_folder),rec_weights_all)
np.save("{}/rec_ctx_weights.npy".format(op_data_folder),rec_ctx_weights_all)
np.save("{}/FR_history.npy".format(op_data_folder),FR_history_all)
np.save("{}/FR_history_ctx.npy".format(op_data_folder),FR_ctx_history_all)
np.save("{}/EX_history.npy".format(op_data_folder),EX_history_all)
np.save("{}/EX_history_ctx.npy".format(op_data_folder),EX_history_ctx_all)
np.save("{}/last_activity.npy".format(op_data_folder),last_activity_all)
np.save("{}/last_activity_ctx.npy".format(op_data_folder),last_activity_ctx_all)
# np.save("./data/Reimagined3/input_history.npy",input_history_all)  
sim_params = {
    "n": n,
    "n_inp": n_inp,
    "n_ctx": n_ctx,
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
            "n_MTL": nn.n_MTL,
            "tau": nn.tau,
            "dt": nn.dt,
            "lr": nn.lr,
            "decay_r": nn.decay_r,
            "lr_ctx": nn.lr_ctx,
            "decay_r_ctx": nn.decay_r_ctx,
            "I0": nn.I0,
            "I1": nn.I1,
            "I2": nn.I2,
            # "excitability": model.excitability.detach().cpu().numpy().tolist(),
        },
        "simulation_params": sim_params
    }

filename = "{}/all_params.json".format(op_data_folder)

with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    # print(f"All parameters saved to {filename}")
plot_corr_matrix(last_activity, fname="{}/corr_matrix.svg".format(op_plot_folder))
plot_corr_matrix(last_activity_ctx, fname="{}/corr_matrix_ctx.svg".format(op_plot_folder))

FR_history_th = (FR_history_all > threshold).astype(float)*FR_history_all
FR_history_th_ctx = (FR_ctx_history_all > threshold).astype(float)*FR_ctx_history_all
breakpoint()
plot_activity_n_excitability_time([FR_history_th[0].T,EX_history_all[0].T],
                       titles=['Neuronal Activity',
                                "Neuronal Excitability"],
                       fname="{}/Activity_n_excitability.svg".format(op_plot_folder),
                       cmaps=['Blues', 'Greens'])

plot_activity_n_excitability_time([FR_history_th_ctx[0].T,EX_history_ctx_all[0].T],
                       titles=['Neuronal Activity',
                                "Neuronal Excitability"],
                       fname="{}/Activity_n_excitability_ctx.svg".format(op_plot_folder),
                       cmaps=['Blues', 'Greens'])
labs = ["FC"] + [f"Off {i+1}" for i in range(N_off_days)]
plot_weights_over_time(rec_weights_all[0],
                       titles=  labs,
                       fname="{}/Rec_w.svg".format(op_plot_folder),
                       cmaps='gray_r')

plot_weights_over_time(rec_ctx_weights_all[0],
                       titles=  labs,
                       fname="{}/Rec_w_ctx.svg".format(op_plot_folder),
                       cmaps='gray_r')

# cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
# xlabs = ["Off 1","Off 2","Off 3","Off 4","Off 5","Recall"]
# Title = "Ensemble similarity of encoding and offline + recall"
# # plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined3/encoding_corr.svg", use_bar_plot=True)
# mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
#     last_activity_all,                # shape: (sims, time, neurons)
#     ref_time_idx=0,         # Encoding
#     xlabels=xlabs,         # must match number of non-ref times
#     include_ref_bar=False,
#     title="Encoding vs. others (mean ± SD across sims)",
#     fname="./plots/Reimagined3/encoding_vs_others_mean_std.svg",
#     bar_colors = cbars[1:last_activity_all.shape[0]+1]
# )

# xlabs = ["Encoding", "Off 1","Off 2","Off 3","Off 4","Off 5"]
# Title = "Ensemble similarity of recall and offline + encoding"
# # plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined3//Recall_corr.svg", use_bar_plot=True)
# last_activity_all, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
#     last_activity_all,                # shape: (sims, time, neurons)
#     ref_time_idx=-1,         # Encoding
#     xlabels=xlabs,         # must match number of non-ref times
#     include_ref_bar=False,
#     title="Encoding vs. others (mean ± SD across sims)",
#     fname="./plots/Reimagined3/recall_vs_others_mean_std.svg",
#     bar_colors = cbars[:last_activity_all.shape[0]]

# )

