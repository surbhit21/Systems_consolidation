import matplotlib.pyplot as plt
import json
import numpy as np
from plotting_widget import *
import torch 
import torch.nn.functional as F
from Utilities import average_firing_rates_with_active,ensamble_overlap
from tqdm import trange
start_seed = 0
torch.manual_seed(start_seed)
class twolayer_FF:
    def __init__(self, n_inp, n_MTL,n_CTX,baseline_e,base_e_ctx,tau=20.0, dt=1.0, act=F.softplus, 
                 lr=1/800, decay_r=1/1000, 
                 lr_ctx = 1/1200, decay_r_ctx = 1e-5,
                 lr_op = 1/1000,
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
        self.lr_op = lr_op
        self.I0 = I0
        self.I1 = I1
        self.I2 = I2
        self.excitability = baseline_e
        self.excitability_ctx = base_e_ctx
        self.plas_threshold = 0
        self.act_threshold = 0#torch.abs(torch.normal(0,0.7,size=(n_MTL,)))
        self.act_threshold_ctx = 0# torch.abs(torch.normal(0,0.7,size=(n_ctx,)))
        self.threshold = threshold
        self.FF_MTL_CTX = 0.#torch.zeros(self.n_CTX)
        self.FB_CTX_MTL = 0.#torch.zeros(self.n_MTL)
        self.gain_ctx = 1.
        self.gain_hpc = 1.
        self.op_neuron = 1
        self.on = 1.0  # plasticity on/off
        self.on_ctx = 1.0
        # Initialize random input weights
        self.input_w = torch.abs(torch.normal(0,0.05,size=(n_inp,n_MTL)))
        # self.input_w = torch.clamp(self.input_w, 0.0, 0.1)
        self.rec_w = torch.zeros(n_MTL, n_MTL)
        self.rec_w_ctx = torch.zeros(self.n_CTX, self.n_CTX)
        
        self.mtl_op_w = torch.abs(torch.normal(0,0.05,size=(self.op_neuron,n_MTL)))
        self.ctx_op_w = torch.abs(torch.normal(0,0.05,size=(self.op_neuron,n_CTX)))
        # Zero initial rate state
        self.rates_ctx = torch.zeros(self.n_CTX)
        self.rates = torch.zeros(n_MTL)
        self.op_rate = torch.zeros(self.op_neuron)
        # self.
    def TurnOFF_FB(self):
        self.FB_CTX_MTL = 0.#torch.zeros_like(self.FB_CTX_MTL)
    def TurnON_FB(self):
        self.FB_CTX_MTL = 0.17#*torch.ones_like(self.FB_CTX_MTL)
    def TurnOFF_FF(self):
        self.FF_MTL_CTX = 0.#torch.zeros_like(self.FF_MTL_CTX)
    def TurnON_FF(self):
        self.FF_MTL_CTX = 2.#*torch.ones_like(self.FF_MTL_CTX)

    def step(self, input_FR, op_signal = 0):
        """
        Perform one timestep of rate dynamics:
        input_vector: shape [n_MTL]
        """
        # calculating the input to the RNN
        self.rates *= (self.rates > 1e-5).float()  # Ensure rates are non-negative
        self.rates_ctx *= (self.rates_ctx > 1e-5).float()
        input_vector = input_FR + self.rec_w @ self.rates - self.FB_CTX_MTL * self.rates_ctx
        input_CTX = input_FR + self.rec_w_ctx @ self.rates_ctx + self.FF_MTL_CTX * self.rates
        input_op = self.mtl_op_w @ self.rates + self.ctx_op_w @ self.rates_ctx

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
        dr_dt = (-self.rates +   self.act(self.gain_hpc * (self.excitability + input_current + self.act_threshold))) / self.tau
        dr_ctx_dt = (-self.rates_ctx +   self.act( self.gain_ctx * (self.excitability_ctx + input_ctx  + self.act_threshold_ctx))) / self.tau

        dr_op_dt = (-self.op_rate + self.act(input_op)) / self.tau

        
        self.rates += (dr_dt * self.dt)
        self.rates_ctx += (dr_ctx_dt * self.dt)
        self.op_rate += (dr_op_dt*self.dt)

        # print(self.rates.max(),self.rates_ctx.max())
        # post_mask = (self.rates > self.threshold).float()
        post_mask = self.rates > self.plas_threshold
        # hebbian plasticity in RNN weights
        hebbian_dw = self.on * self.lr * torch.outer(self.rates*post_mask, self.rates*post_mask) * self.dt
        decay = self.decay_r * self.rec_w * self.dt


        # hebbian plasticity in input weights
        self.rec_w += (hebbian_dw - decay)

        hebb_mtl_op = self.lr_op * op_signal * torch.outer(self.op_rate, self.rates*post_mask) * self.dt


        
        self.mtl_op_w += hebb_mtl_op
        sum_w = torch.sum(self.mtl_op_w)
        self.mtl_op_w /= sum_w
        self.mtl_op_w = torch.clamp(self.mtl_op_w,min=0)


        post_mask = self.rates_ctx > self.plas_threshold
        hebbian_dw_ctx = self.on_ctx * self.lr_ctx * torch.outer(self.rates_ctx*post_mask, self.rates_ctx*post_mask) * self.dt
        decay_ctx = self.decay_r_ctx * self.rec_w_ctx * self.dt
        self.rec_w_ctx += (hebbian_dw_ctx - decay_ctx)
        
        hebb_ctx_op = self.lr_op * op_signal * torch.outer(self.op_rate, self.rates_ctx*post_mask) * self.dt

        self.ctx_op_w += hebb_ctx_op
        sum_w = torch.sum(self.ctx_op_w)
        self.ctx_op_w /= sum_w
        self.ctx_op_w = torch.clamp(self.ctx_op_w,min=0)
        
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
        
        return self.rates.detach().clone(),self.rates_ctx.detach().clone(), self.op_rate.detach().clone()
    
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



N_off_days = 11
n = 10 + (N_off_days)*20 #10 default + 20 neurons per off day
n_inp = n 
n_ctx = n 
E_fl = 2.2
E_fl_ctx = 2.2
max_e = 3
E_ref = 0.7
threshold = 2
off_set = 0



# base_E[:off_set] += 2
sim_name = "HPC_act_block"
notes = "trying with Feedback and feedforward connection of equal strength"
FC_inp = 18
input = FC_inp*torch.ones(n_inp)
zero_input = torch.zeros(n_inp)
mu_ex = 0
sigma_ex = 1

ID = 1000
dt = 1
NUM_SIM = 10
t_off = 100
IR = 100
Nrep = 10
total_time = 0
FR_history_all = []
EX_history_all = []
EX_history_ctx_all = []
rec_weights_all = []
ff_weights_all = []
mtl_op_weights_all =[]
ctx_op_weights_all = []
last_activity_all = []
input_history_all = []
rec_ctx_weights_all = []
FR_ctx_history_all = []
last_activity_ctx_all = []
FR_op_history_all = []
dob = [9]
t_encoding = 1000




for i in trange(NUM_SIM):
    total_time = 0
    torch.manual_seed(start_seed + i)
    base_E = torch.abs(torch.normal(mu_ex,sigma_ex,size=(n,)))
    base_e_ctx = torch.abs(torch.normal(mu_ex,sigma_ex,size=(n_ctx,)))
    nn = twolayer_FF(n_inp=n_inp, n_MTL=n,n_CTX=n_ctx, baseline_e = base_E.clone(),
                     base_e_ctx=base_e_ctx.clone(), tau=20.0, dt=dt, act=torch.relu,
                    lr=1/1000, decay_r=1/1200, 
                    lr_ctx = 5e-5,decay_r_ctx=5e-7,
                    lr_op = 1e-3,
                    I0=7, I1=0.7, I2=0.04)
    FR_history = []
    FR_history_ctx = []
    FR_op_history = []
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
    mtl_op_weights = []
    ctx_op_weights = []
    # total_time += 10
    nn.excitability = base_E.clone()
    day = 0
    nn.excitability[off_set+(day)*20:off_set+(day)*20+20] += E_fl
    nn.excitability = torch.clip(nn.excitability,0,max_e)
    nn.excitability_ctx  = base_e_ctx.clone()
    nn.excitability_ctx[(off_set+(day)*20): (off_set+(day)*20+20)] += E_fl_ctx
    nn.excitability_ctx = torch.clip(nn.excitability_ctx,0,max_e)
    # for t in range(10):
    #     next_FR,FR_ctx,FR_op = nn.step(zero_input,0)
    #     FR_history.append(next_FR)
    #     FR_history_ctx.append(FR_ctx)
    #     FR_op_history.append(FR_op)
    #     EX_history.append(nn.excitability.detach().clone().numpy())
    #     EX_history_ctx.append(nn.excitability_ctx.detach().clone().numpy())
    #     input_history.append(zero_input.numpy())
    # total_time += t_encoding
    # for t in range(t_encoding):
    #     next_FR,FR_ctx,FR_op = nn.step(input,1.0)
    #     FR_history.append(next_FR)
    #     FR_history_ctx.append(FR_ctx)
    #     FR_op_history.append(FR_op)
    #     EX_history.append(nn.excitability.detach().clone().numpy())
    #     EX_history_ctx.append(nn.excitability_ctx.detach().clone().numpy())
    #     input_history.append(zero_input.numpy())
    total_time += ID
    for t in range(ID):
        next_FR,FR_ctx,FR_op = nn.step(zero_input,0)
        FR_history.append(next_FR)
        FR_history_ctx.append(FR_ctx)
        FR_op_history.append(FR_op)
        EX_history.append(nn.excitability.detach().clone().numpy())
        EX_history_ctx.append(nn.excitability_ctx.detach().clone().numpy())
        input_history.append(zero_input.numpy())
    for day in range(N_off_days):
        day_activity = []
        day_activity_ctx = []

        # torch.manual_seed(120+day)
        # base_E = torch.abs(torch.normal(mu_ex,sigma_ex,size=(n,)))
        # base_e_ctx = torch.abs(torch.normal(mu_ex,sigma_ex,size=(n_ctx,)))
        nn.excitability = base_E.clone()
        nn.excitability[off_set+(day)*20:off_set+(day)*20+20] += E_fl
        nn.excitability = torch.clip(nn.excitability,0,max_e)
        nn.excitability_ctx  = base_e_ctx.clone()
        nn.excitability_ctx[(off_set+(day)*20):(off_set+(day)*20 + 20)] += E_fl_ctx
        nn.excitability_ctx = torch.clip(nn.excitability_ctx,0,max_e)
        inp_to_network = input                
        # if day == 1:
        #     nn.TurnOFF_FB()
        #     nn.TurnOFF_FF()
        # else:
        #     nn.TurnON_FB()
        #     nn.TurnON_FF()
        if day == 0:
            op_learning = 1.
        else:
            op_learning = 0.0
            
        for rep in range(Nrep):
            nn.TurnON_FB()
            nn.TurnON_FF()
            
            if day in dob:
                nn.gain_hpc = 0.0
                # nn.TurnOFF_FB()
            else:
                nn.gain_hpc  = 1.0
                # nn.TurnON_FB()
            
            total_time += t_off
            for t in range(t_off):
                next_FR,FR_ctx,FR_op = nn.step(inp_to_network,op_learning)
                FR_history.append(next_FR)
                FR_history_ctx.append(FR_ctx)
                FR_op_history.append(FR_op)
                EX_history.append(nn.excitability.detach().clone().numpy())
                EX_history_ctx.append(nn.excitability_ctx.detach().clone().numpy())
                input_history.append(input.numpy())
            day_activity.append(np.mean(FR_history[-t_off:],axis=0))
            day_activity_ctx.append(np.mean(FR_history_ctx[-t_off:],axis=0))    
            total_time += IR
            nn.TurnOFF_FB()
            nn.TurnOFF_FF()
            for t in range(IR):
                next_FR,FR_ctx, FR_op = nn.step(zero_input,0)
                FR_history.append(next_FR)
                FR_history_ctx.append(FR_ctx)
                FR_op_history.append(FR_op)
                EX_history.append(nn.excitability.detach().clone().numpy())
                EX_history_ctx.append(nn.excitability_ctx.detach().clone().numpy())
                input_history.append(zero_input.numpy())
        rep_Activity.append(day_activity)
        rec_weights.append(nn.rec_w.detach().clone().numpy())
        rec_ctx_weights.append(nn.rec_w_ctx.detach().clone().numpy())
        mtl_op_weights.append(nn.mtl_op_w.detach().clone().numpy())
        ctx_op_weights.append(nn.ctx_op_w.detach().clone().numpy())
        # ff_weights.append(nn.input_w.detach().clone().numpy())
        last_activity.append(np.mean(day_activity,axis=0))
        last_activity_ctx.append(np.mean(day_activity_ctx,axis=0))
        total_time += ID
        for t in range(ID):
            next_FR,FR_ctx,FR_op = nn.step(zero_input,0)
            FR_history.append(next_FR)
            FR_history_ctx.append(FR_ctx)
            FR_op_history.append(FR_op)
            EX_history.append(nn.excitability.detach().clone().numpy())
            EX_history_ctx.append(nn.excitability_ctx.detach().clone().numpy())
            input_history.append(zero_input.numpy())
    # breakpoint()
    FR_history_all.append(FR_history)
    FR_ctx_history_all.append(FR_history_ctx)
    FR_op_history_all.append(FR_op_history)
    EX_history_all.append(EX_history)
    EX_history_ctx_all.append(EX_history_ctx)
    rec_weights_all.append(rec_weights)
    rec_ctx_weights_all.append(rec_ctx_weights)
    ff_weights_all.append(ff_weights)
    last_activity_all.append(last_activity)
    input_history_all.append(input_history)
    last_activity_ctx_all.append(last_activity_ctx)
    mtl_op_weights_all.append(mtl_op_weights)
    ctx_op_weights_all.append(ctx_op_weights)
FR_history_all = np.stack(FR_history_all)
FR_ctx_history_all = np.stack(FR_ctx_history_all)
FR_op_history_all  = np.stack(FR_op_history_all)
# input_history = np.stack(input_history)


EX_history_all = np.stack(EX_history_all)
EX_history_ctx_all = np.stack(EX_history_ctx_all)
last_activity_all = np.stack(last_activity_all)
last_activity_ctx_all = np.stack(last_activity_ctx_all)
rec_weights_all = np.stack(rec_weights_all)
rec_ctx_weights_all = np.stack(rec_ctx_weights_all)

op_data_folder = "./data/{}".format(sim_name)
op_plot_folder = "./plots/{}".format(sim_name)# --- IGNORE ---
os.makedirs(op_data_folder, exist_ok=True)
np.save("{}/rec_weights.npy".format(op_data_folder),rec_weights_all)
np.save("{}/mtl_op_weights.npy".format(op_data_folder),mtl_op_weights_all)
np.save("{}/ctx_op_weights.npy".format(op_data_folder),ctx_op_weights_all)
np.save("{}/rec_ctx_weights.npy".format(op_data_folder),rec_ctx_weights_all)
np.save("{}/FR_history.npy".format(op_data_folder),FR_history_all)
np.save("{}/FR_history_ctx.npy".format(op_data_folder),FR_ctx_history_all)
np.save("{}/FR_history_op.npy".format(op_data_folder),FR_op_history_all)
np.save("{}/EX_history.npy".format(op_data_folder),EX_history_all)
np.save("{}/EX_history_ctx.npy".format(op_data_folder),EX_history_ctx_all)
np.save("{}/last_activity.npy".format(op_data_folder),last_activity_all)
np.save("{}/last_activity_ctx.npy".format(op_data_folder),last_activity_ctx_all)

# np.save("./data/Reimagined3/input_history.npy",input_history_all)  
nn.TurnON_FB()
nn.TurnON_FF()
sim_params = {
    "n": n,
    "n_inp": n_inp,
    "n_ctx": n_ctx,
    "E_fl": E_fl,
    "FC_inp":FC_inp,
    "E_fl_ctx":E_fl_ctx,
    "threshold": threshold,
    "ID": ID,
    "N_off_days": N_off_days,
    "t_off": t_off,
    "IR": IR,
    "Nrep": Nrep,
    "start_seed": start_seed,  # if you want reproducibility
    "max_e":max_e,
    "total_time": total_time,
    "dt": dt,
    "NUM_SIM": NUM_SIM,
}
data = {
        "model_params": {
            "dop":dob,
            "n_MTL": nn.n_MTL,
            "tau": nn.tau,
            "dt": nn.dt,
            "lr": nn.lr,
            "decay_r": nn.decay_r,
            "lr_ctx": nn.lr_ctx,
            "decay_r_ctx": nn.decay_r_ctx,
            "lr_op":nn.lr_op,
            "I0": nn.I0,
            "I1": nn.I1,
            "I2": nn.I2,
            "mu_ex":mu_ex,
            "sigma_ex":sigma_ex,
            "ff":nn.FF_MTL_CTX,#.detach().clone().tolist(),
            "fb":nn.FB_CTX_MTL#.detach().clone().tolist()
        },
        "simulation_params": sim_params
    }

filename = "{}/all_params.json".format(op_data_folder)

with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    # print(f"All parameters saved to {filename}")
last_activity_all=  (last_activity_all > threshold).astype(float)*last_activity_all
last_activity_ctx_all =  (last_activity_ctx_all > threshold).astype(float)*last_activity_ctx_all
plot_corr_matrix(last_activity_all[0], fname="{}/corr_matrix.svg".format(op_plot_folder))
plot_corr_matrix(last_activity_ctx_all[0], fname="{}/corr_matrix_ctx.svg".format(op_plot_folder))
# plt.plot()

cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
xlabs = ["Day 0"] + [f"Off {i+1}" for i in range(N_off_days-1)]
Title = "Ensemble similarity"
# plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr.svg", use_bar_plot=True)
mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    last_activity_all ,                # shape: (sims, time, neurons)
    ref_time_idx=0,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=True,
    title="Cell population \n activity correlation",
    fname="{}/encoding_vs_others_mean_std.svg".format(op_plot_folder),
    cmap = "Oranges",
    marker = "^"
)
# breakpoint()
mean_corr_cxt, std_corr_cxt, per_sim_corr_cxt, idx_cxt = plot_mean_std_corr_over_time(
    last_activity_ctx_all ,                # shape: (sims, time, neurons)
    ref_time_idx=0,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=True,
    title="Cell population \n activity correlation",
    fname="{}/encoding_vs_others_mean_std_ctx.svg".format(op_plot_folder),
    cmap = "Greens"
)


FR_history_th = (FR_history_all > threshold).astype(float)*FR_history_all
FR_history_th_ctx = (FR_ctx_history_all > threshold).astype(float)*FR_ctx_history_all

breakpoint()
timepoints = np.arange(0,total_time,1)*1
plt.plot(timepoints,FR_op_history_all[0,:,0],label="OP neuron FR")
plt.legend()
plt.show()

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

labs = ["Day 0"] + [f"Day {i+1}" for i in range(N_off_days)]
plot_weights_over_time(rec_weights_all[0],
                       titles=  labs,
                       fname="{}/Rec_w.svg".format(op_plot_folder),
                       cmaps='gray_r')

plot_weights_over_time(rec_ctx_weights_all[0],
                       titles=  labs,
                       fname="{}/Rec_w_ctx.svg".format(op_plot_folder),
                       cmaps='gray_r')

plot_weights_over_time(mtl_op_weights_all[0],
                       titles=  labs,
                       fname="{}/mtl_op_w.svg".format(op_plot_folder),
                       cmaps='gray_r')
plot_weights_over_time(ctx_op_weights_all[0],
                       titles=  labs,
                       fname="{}/ctx_op_w.svg".format(op_plot_folder),
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

