import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from plotting_widget import *
from Utilities import average_firing_rates_with_active, ensamble_overlap, get_tagged_neurons
import random
# ---------- helpers ----------
def relu(x):
    return np.maximum(x, 0.0)

# ---------- model ----------
class twolayer_FF:
    def __init__(self, n_neurons, tau=20.0, dt=1.0, 
                 tau_lr0=800, tau_decay=1000, I0=7, I1=0.5, I2=0.05):
        # network and simulation properties
        self.n_neurons = n_neurons
        self.dt = 1

        # neuronal properties
        self.tau = float(tau)
        # self.act = act

        # excitability params
        self.sigma = 1
        e0 =  np.abs(np.random.normal(0.0,self.sigma, size=(self.n_neurons, 1)))
        self.epsi_i0 = e0.copy()
        self.epsi_i = e0.copy()
        self.beta_e = 0.9
        self.tau_e = 10
        

        # epigenetic priming variable
        self.tau_alpha_p = 10
        self.tau_alpha_m = 1000
        self.alpha = np.zeros((n_neurons,1), dtype=float)
        self.theta = 2

        # synaptic variables
        self.tau_lr_0 = tau_lr0
        self.tau_lr_i = tau_lr0#np.full(n_neurons, float(tau_lr0), dtype=float)
        self.e_tau_decay = 3500
        self.tau_decay = tau_decay
        # self.beta_w = 0.9

        # global inhibition
        self.I0 = float(I0)
        self.I1 = float(I1)
        self.I2 = float(I2)

        self.scaleE = 0
        self.scaleW = 0
        self.base = 0.0
        # tagged time (first t at which the FR is > theta for encoding and for recall)
        self.tag_t = np.zeros((n_neurons,1), dtype=float)
        self.tag_r = np.zeros((n_neurons,1), dtype=float)

        self.max_w = 1.0
        self.noise_scale = 0
        self.rec_w = np.zeros((n_neurons, n_neurons), dtype=float)
        self.rates = np.zeros((n_neurons,1), dtype=float)

        # synaptic pruning parameters
        self.prune_threshold = 0.1
        self.prune_prob =0.2
        self.M = np.ones((n_neurons, n_neurons), dtype=float)

        self.tag_ij = np.zeros((self.n_neurons,self.n_neurons), dtype=float)
        self.tag_theta = 3e-3

    def prune_below_threshold_prob(self, threshold=0.05, p_prune=0.5):
        """
        If a synaptic weight falls below `threshold`, it has probability `p_prune`
        of being permanently pruned (zeroed and masked).
        """
        # synapses that are currently present and below threshold
        weak = (self.rec_w < threshold) & (self.M == 1)

        if not np.any(weak):
            return 0  # nothing to prune

        # generate random numbers for pruning decisions
        rand_mask = np.random.rand(*self.rec_w.shape)

        # prune where weight is weak *and* random draw < p_prune
        to_prune = weak & (rand_mask < p_prune)

        # apply pruning
        self.M[to_prune] = 0.0
        self.rec_w[to_prune] = 0.0


        
        return int(np.sum(to_prune))  # return number of pruned synapses

    def check_LateLTP(self):
        """
        Check which neurons have crossed the late LTP threshold and set is_long_term accordingly.
        """
        late_LTP_threshold = 1.5
        self.is_long_term = (self.alpha >= late_LTP_threshold).astype(float)

    def step(self,ct,ctx1,ctx2,seq1,seq2,seed = 0):
        """
        Perform one timestep of rate dynamics:
        input_FR: shape [n_neurons] or scalar
        """
        # total synaptic input
        input_FR = self.INPUT(ct,ctx1,seq1)[np.newaxis].T + self.INPUT(ct,ctx2,seq2)[np.newaxis].T
        if np.isscalar(input_FR):
            input_FR = np.full((self.n_neurons, 1), float(input_FR), dtype=float)
        else:
            input_FR = np.asarray(input_FR, dtype=float)
            if input_FR.shape == (self.n_neurons,):
                input_FR = input_FR.reshape(self.n_neurons, 1)
            assert input_FR.shape == (self.n_neurons, 1)
        # breakpoint()
        

        input_vector = input_FR + self.rec_w.dot(self.rates)
        self.rates = self.rates*(self.rates>1e-5)
        # global inhibition
        if np.isnan(self.rec_w).any():
            breakpoint()
            print("rec_w max >",self.rec_w.max(), ct)
        r = self.rates
        sum_r = float(np.sum(r))                           # scalar
        sum_r_pos = np.sum(np.multiply(np.maximum(0,r),r))
        I_inhib = self.I0 + self.I1 * sum_r + self.I2 * sum_r_pos

        if ct == 0:
            np.random.seed(seed)
        e0 = np.zeros(self.n_neurons)
        e0 = np.random.normal(0.0, self.sigma, size=(self.n_neurons, 1))
        self.epsi_i0 = np.sqrt(e0*e0)
        if ct == 0:
            self.epsi_i = self.epsi_i0.copy()
        # total input current

        input_current = input_vector - I_inhib
        if ct in seq1:
            print("input current seq1",ct,input_current.max(),I_inhib,(self.rec_w @ self.rates).max(),self.INPUT(ct,ctx1,seq1)[np.newaxis].T .max(),self.INPUT(ct,ctx2,seq2)[np.newaxis].T.max())
        # rate dynamics
        dr_dt = (-self.rates + np.maximum(0,self.epsi_i + input_current)) / self.tau

        # hebbian plasticity in recurrent weights
        op_prod = np.outer(self.rates, self.rates)
        w_noise = np.random.normal(0.0, 0.001, size=(self.n_neurons, self.n_neurons))
        dWdt = (1 + self.scaleW * self.alpha) * (self.rates >= self.theta) * self.rates.dot(self.rates.T) / self.tau_lr_0  + (self.rates < self.theta) * self.rates.dot(self.rates.T) / self.tau_lr_0  #+ (self.noise_scale*w_noise)
        

        dWdt -= (1.0 / (((1.0-self.tag_ij) * self.e_tau_decay) + (self.tag_ij * self.tau_decay))) * self.rec_w
        self.tag_ij = (dWdt > self.tag_theta).astype(float) + self.tag_ij
        self.tag_ij = np.clip(self.tag_ij, 0.0, 1.0)
        # breakpoint()
        # if 2433 < ct < 2438:
        #     breakpoint()
        #     print("op_prod",op_prod.shape)
        dWdt = np.multiply(dWdt,np.logical_not(np.logical_or(np.logical_and(self.rec_w>=self.max_w,dWdt>0),np.logical_and(self.rec_w<=0,dWdt<0))))
        # dw_dt = dWdt + self.noise_scale*noise
        # gate_post = (self.rates >= self.theta).astype(float)[:, None]          # (N,1)
        # alpha_post = (1.0 + self.scaleW * self.alpha)[:, None]                 # (N,1)
        # hebb_core = np.outer(self.rates, self.rates)                           # (N,N)
        # base_lr = (1.0 / self.tau_lr_i)[:, None]                               # (N,1) postsyn learning rate

        # hebbian_dw = (gate_post * alpha_post + (1.0 - gate_post)) * base_lr * hebb_core * self.dt
        # hebbian_dw =(1.0 / self.tau_lr_i)[:, None] * np.outer(self.rates, self.rates) * self.dt
        # decay = (1.0 / self.tau_decay) * self.rec_w * self.dt

        # change in priming variable
        # breakpoint()
        # self.check_LateLTP()
        dalpha_dt = np.zeros((self.n_neurons,1))
        dalpha_dt += (self.tag_t !=0) * (ct >= 3000) * (ct<= 3100) * (self.tag_t < 3000) * (2 - self.alpha) / self.tau_alpha_p
        dalpha_dt += (self.tag_t !=0) * (ct >= 6000) * (ct<= 6100) * (self.tag_t > 3000)* (self.tag_t < 6000)*(2 - self.alpha) / self.tau_alpha_p
        dalpha_dt += (self.tag_t !=0) * (ct > 3100) * (self.tag_t < 3000) * (1 - self.alpha) / self.tau_alpha_m
        dalpha_dt += (self.tag_t !=0) * (ct > 6100) * (self.tag_t > 3000)* (self.tag_t < 6000) * (1 - self.alpha) / self.tau_alpha_m

        dexcdt = np.zeros((self.n_neurons,1))
        dexcdt = ((self.rates > self.theta) * (self.scaleE * self.alpha + self.epsi_i0 - self.epsi_i ) / self.tau_e) + ((self.rates < self.theta) * (self.epsi_i0 - self.epsi_i) / self.tau_e)
        # if dexcdt.max() != 0 and self.rates.max()>1e-5:
        #     # breakpoint()
        #     print("ex changed",dexcdt.max(),ct,self.rates.max())
        self.tag_t = self.tag_t + (self.rates > self.theta).astype(float) * (self.tag_t == 0).astype(float) * ct
        if ct > 25000:
            self.tag_r = self.tag_r + (self.rates > self.theta).astype(float) * (self.tag_r == 0).astype(float) * ct
        # if (self.rates > self.theta).all():
        #     breakpoint()
        # breakpoint()
        self.rates += dr_dt * self.dt
        self.rec_w += dWdt * self.dt
        np.clip(self.rec_w, 0.0, self.max_w, out=self.rec_w)
        
        self.alpha += dalpha_dt*self.dt
        self.epsi_i += dexcdt*self.dt

        # clamp weights non-negative and <= 1
        return self.rates[:,0].copy(),input_FR[:,0]

    def INPUT(self,ct1,ctx,seq,context_inp=16):
        L = 0
        pol = 1
        for step in seq:
            L += np.tanh(ct1-step)*pol
            pol *= -1
        return (context_inp-self.base) * ctx * L/2 + self.base

# ---------- simulation ----------
start_seed = 0
np.random.seed(start_seed)

n1 = 50
threshold = 2
# FC_inp = 25
# cip = 16 # scalar input (broadcast)
# off_context_inp = 1.6
zero_input = np.zeros(n1, dtype=float)
# color = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51"]
color = ["#001219","#005f73","#0a9396","#94d2bd","#e9d8a6","#ee9b00","#ca6702","#bb3e03","#ae2012","#9b2226"]
ID = 1000
T = 30000
NUM_SIM = 1
N_off_days = 1
t_off = 100
IR = 100
Nrep = 10
t_Recall_start = 8000
t_recall = 100
IR_recall = 6000 - t_recall
N_recall = 4
overlap = 5/100
seqA1 = []
# ctxA = np.zeros((1,n1))
ctxA = np.array([0., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 1.,\
                 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.,
                 1., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 1.])
# ctxA[:int(n1/2)] = 1
# # random.seed(0)
# random.shuffle(ctxA)
#ctxB = np.zeros(N)
#ctxB[:int(overlap*N)] = ctxA[:int(overlap*N)]
#ctxB[int(overlap*N):] = 1 - ctxA[int(overlap*N):]
ov = np.arange(0,n1)
random.shuffle(ov)
# ctxB = 1 - ctxA
# for n in ov[:int(overlap*n1)]:
#     ctxB[n] = ctxA[n]
seqA = [1000,
  1100,
  1200,
  1300,
  1400,
  1500,
  1600,
  1700,
  1800,
  1900,
  2000,
  2100,
  2200,
  2300,
  2400,
  2500,
  2600,
  2700,
  2800,
  2900,
  8000,
  8100,
  14000,
  14100,
  20000,
  20100,
  26000,
  26100,
  32000,
  32100]
seqB = []
ctxB = np.array([1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., \
                  1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., \
                1., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0. ,0., 1., 1., 0.])
breakpoint()
exp_condition = "early_late_ltp"

FR_history_all_cnt = []
EX_history_all_cnt = []
rec_weights_all_cnt = []
ff_weights_all_cnt = []
last_activity_all_cnt = []
input_history_all_cnt = []
alpha_history_all_cnt = []

FR_history_all_res = []
EX_history_all_res = []
rec_weights_all_res = []
ff_weights_all_res = []
last_activity_all_res = []
input_history_all_res = []
alpha_history_all_res = []


nn_cnt = twolayer_FF(n_neurons=n1, tau=20, dt=1.0, tau_lr0=2500, tau_decay=4300, I0=7, I1=0.5, I2=0.05)
nn_res = twolayer_FF(n_neurons=n1, tau=20, dt=1.0, tau_lr0=2500, tau_decay=4300, I0=7, I1=0.5, I2=0.05)  


# nn_cnt.noise_scale = 0
# nn.max_w = 0.9
# nn_cnt.max_w = 1

for current_t in range(0,T):
    next_FR,input_fr  = nn_cnt.step(current_t,ctxA,ctxB,seqA,seqB)
    FR_history_all_cnt.append(next_FR.copy())
    input_history_all_cnt.append(input_fr.copy())
    EX_history_all_cnt.append(nn_cnt.epsi_i[:,0].copy())
    alpha_history_all_cnt.append(nn_cnt.alpha[:,0].copy())
    # if current_t in [2900,8100,14100,20100,26100]:
    rec_weights_all_cnt.append(nn_cnt.rec_w.copy())

# nn_res.noise_scale = 0
# nn_res.max_w = 1
nn_res.scaleE = 0.9
nn_res.scaleW = 0.9
for current_t in range(0,T):
    next_FR,input_fr  = nn_res.step(current_t,ctxA,ctxB,seqA,seqB)
    FR_history_all_res.append(next_FR.copy())
    input_history_all_res.append(input_fr.copy())
    EX_history_all_res.append(nn_res.epsi_i[:,0].copy())
    alpha_history_all_res.append(nn_res.alpha[:,0].copy())
    # if current_t in [2900,8100,14100,20100,26100]:
    rec_weights_all_res.append(nn_res.rec_w.copy())

print(seqA1, seqA)
print("current_t == 32000", current_t == T)
FR_history_all_cnt = np.stack(FR_history_all_cnt)
EX_history_all_cnt = np.stack(EX_history_all_cnt)
alpha_history_all_cnt = np.stack(alpha_history_all_cnt)
input_history_all_cnt = np.stack(input_history_all_cnt)
rec_weights_all_cnt = np.stack(rec_weights_all_cnt)

FR_history_all_res = np.stack(FR_history_all_res)
EX_history_all_res = np.stack(EX_history_all_res)
alpha_history_all_res = np.stack(alpha_history_all_res)
input_history_all_res = np.stack(input_history_all_res)
rec_weights_all_res = np.stack(rec_weights_all_res)
# breakpoint()

inp_data_folder = "./data/epigenetic_paper"
op_plot_folder = "./plots/epigenetic_paper"
os.makedirs(inp_data_folder, exist_ok=True)
os.makedirs(op_plot_folder, exist_ok=True)
# np.save(f"{inp_data_folder}/FR_history.npy", FR_history_all_cnt)
# np.save(f"{inp_data_folder}/EX_history.npy", EX_history_all_cnt)
# np.save(f"{inp_data_folder}/alpha_history.npy", alpha_history_all_cnt)
# np.save(f"{inp_data_folder}/rec_W.npy", rec_weights_all_cnt)
# np.save(f"{inp_data_folder}/input_history.npy", input_history_all_cnt)

# sim_params = {
#     "n": n1,
#     "threshold": threshold,
#     "ID": ID,
#     "N_off_days": N_off_days,
#     "t_off": t_off,
#     "IR": IR,
#     "Nrep": Nrep,
#     "start_seed": start_seed
# }
# data = {
#     "model_params": {
#         "n_neurons": nn.n_neurons,
#         "tau": nn.tau,
#         "dt": nn.dt,
#         "Tau_lr": nn.tau_lr_0,
#         "tau_decay_r": nn.tau_decay,
#         "I0": nn.I0,
#         "I1": nn.I1,
#         "I2": nn.I2,
#     },
#     "simulation_params": sim_params
# }
# filename = f"{inp_data_folder}/all_params.json"
# with open(filename, "w") as f:
#     json.dump(data, f, indent=4)

FR_history_th = (FR_history_all_cnt >= 0).astype(float) * FR_history_all_cnt
FR_history_th_res = (FR_history_all_res >= 0).astype(float) * FR_history_all_res
tn1, un1 = get_tagged_neurons(FR_history_all_cnt[:3000,:].T, threshold)
un_cnt = [i for i in range(n1) if i not in tn1]
# tn2, un2 = get_tagged_neurons(FR_history_all_cnt[3000:6000,:].T, threshold)

tn1_res = get_tagged_neurons(FR_history_all_res[:3000,:].T, threshold)[0]
un_res = [i for i in range(n1) if i not in tn1_res]

avg_sw_cnt= rec_weights_all_cnt[:,tn1,tn1].mean(axis=1)
avg_sw_res= rec_weights_all_res[:,tn1_res,tn1_res].mean(axis=1)
# avg_sw = avg_sw_per_neuron.mean(axis=1)
t_points = np.arange(1, current_t +2, 1) * 1e-3
# plt.figure(figsize=(8,4))
fs = 22  # <--- font size for ticks and labels
# im = nn_cnt.rec_w + nn_cnt.tag_ij
# fig = plt.figure(figsize=(8,4))
# gs = gridspec.GridSpec(1, 1 + 1, width_ratios=[1] * 1 + [0.05], wspace=0.3)
# plt.imshow(im)
# cax = fig.add_subplot(gs[0, -1])
# cbar = fig.colorbar(im, cax=cax)

# cbar.ax.tick_params(labelsize=fs)
# fname = f"{op_plot_folder}/tag_w.svg"
# save_plot(fname)
# plt.show()

# im = nn_res.rec_w + nn_res.tag_ij
# fig = plt.figure(figsize=(8,4))
# gs = gridspec.GridSpec(1, 1 + 1, width_ratios=[1] * 1 + [0.05], wspace=0.3)
# plt.imshow(im)
# cax = fig.add_subplot(gs[0, -1])
# cbar = fig.colorbar(im, cax=cax)

# cbar.ax.tick_params(labelsize=fs)
# fname = f"{op_plot_folder}/tag_w_{exp_condition}.svg"
# save_plot(fname)
# plt.show()
# breakpoint()
ims = np.array([(nn_cnt.rec_w > 0.2).astype(float) + nn_cnt.tag_ij, (nn_res.rec_w > 0.2).astype(float)+ nn_res.tag_ij])
labs = ["STC only Model","STC + Epigenetic Priming Model"]
plot_weights_over_time(
    ims,
    titles=labs,
    fname=f"{op_plot_folder}/Rec_w_tag_{exp_condition}_rescue.svg",
    cmaps='viridis'
)

breakpoint()

plt.figure(figsize=(8,4))
plt.plot(t_points, avg_sw_cnt, c=color[8], label='', linestyle='--')
plt.plot(t_points, avg_sw_res, c=color[8], label='', linestyle='-')

h = -0.05
for s in range(int(len(seqA)/2-1)):
    plt.plot([seqA[2*s]*1e-3, seqA[2*s+1]*1e-3], [h, h], 'k')
for s in range(int(len(seqB)/2-1)):
    plt.plot([seqB[2*s]*1e-3, seqB[2*s+1]*1e-3], [h, h], 'k')

plt.hlines(y=0, xmin=0, xmax=t_points[-1], colors='k', linestyles='--')

plt.xlabel('Time steps (x1000)', fontsize=fs)
plt.ylabel('Avg. synap weights \n of tagged neurons', fontsize=fs)
plt.yticks(np.arange(0,1.1,0.2),
           [0,0.2,0.4,0.6,0.8,1.0],
           fontsize=fs)
plt.xticks(fontsize=fs)

fname = f"{op_plot_folder}/avg_sw_{exp_condition}.svg"
save_plot(fname)
plt.show()


# ---- Second Plot ---- #
plt.figure(figsize=(8,4))
for t1 in tn1_res:
    plt.plot(t_points, FR_history_all_res[:, t1], c=color[8])
for t1 in tn1:
    plt.plot(t_points, FR_history_all_cnt[:, t1], c=color[1])

plt.hlines(y=2, xmin=0, xmax=t_points[-1], colors='k', linestyles='--')

plt.ylabel('Firing Rate', fontsize=fs)
plt.xlabel('Time steps (x1000)', fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

h = -0.05
for s in range(int(len(seqA)/2)):
    plt.plot([seqA[2*s]*1e-3, seqA[2*s+1]*1e-3], [h, h], 'k')
for s in range(int(len(seqB)/2)):
    plt.plot([seqB[2*s]*1e-3, seqB[2*s+1]*1e-3], [h, h], 'k')

fname = f"{op_plot_folder}/tagged_cnt_vs_rescue_{exp_condition}.svg"
save_plot(fname)
plt.show()


# breakpoint()  # optional

plot_activity_n_excitability_time(
    [FR_history_th.T, EX_history_all_cnt.T],
    titles=['Neuronal Activity', "Neuronal Excitability"],
    fname=f"{op_plot_folder}/Activity_n_excitability_{exp_condition}.svg",
    cmaps=['binary', 'Blues']
)

plot_activity_n_excitability_time(
    [input_history_all_cnt.T, alpha_history_all_cnt.T],
    titles=['Input', "Alpha"],
    fname=f"{op_plot_folder}/input_and_alpha_{exp_condition}.svg",
    cmaps=['binary', 'Blues']
)

labs = ["Encoding"] + [f"Recall {i+1}" for i in range(N_recall)]
plotting_weights_times = [2900,8100,14100,20100,26100]
plot_weights_over_time(
    rec_weights_all_cnt[plotting_weights_times],
    titles=labs,
    fname=f"{op_plot_folder}/Rec_w_{exp_condition}.svg",
    cmaps='gray_r'
)

plot_activity_n_excitability_time(
    [FR_history_all_res.T, EX_history_all_res.T],
    titles=['Neuronal Activity', "Neuronal Excitability"],
    fname=f"{op_plot_folder}/Activity_n_excitability_{exp_condition}_rescue.svg",
    cmaps=['binary', 'Blues']
)

plot_activity_n_excitability_time(
    [input_history_all_res.T, alpha_history_all_res.T],
    titles=['Input', "Alpha"],
    fname=f"{op_plot_folder}/input_and_alpha_{exp_condition}_rescue.svg",
    cmaps=['binary', 'Blues']
)

labs = ["Encoding"] + [f"Recall {i+1}" for i in range(N_recall)]
plotting_weights_times = [2900,8100,14100,20100,26100]
plot_weights_over_time(
    rec_weights_all_res[plotting_weights_times],
    titles=labs,
    fname=f"{op_plot_folder}/Rec_w_{exp_condition}_rescue.svg",
    cmaps='gray_r'
)


