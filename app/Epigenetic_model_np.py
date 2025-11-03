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
    def __init__(self, n_neurons, tau=20.0, dt=1.0, act=relu,
                 tau_lr0=800, tau_decay=1000, I0=7, I1=0.5, I2=0.05):
        # network and simulation properties
        self.n_neurons = n_neurons
        self.dt = dt

        # neuronal properties
        self.tau = float(tau)
        self.act = act

        # excitability params
        self.sigma = 1
        self.epsi_i0 = np.abs(np.random.normal(0.0,self.sigma,self.n_neurons))
        self.epsi_i = np.abs(np.random.normal(0.,self.sigma,self.n_neurons))
        self.beta_e = 0.9
        self.tau_e = 10
        

        # epigenetic priming variable
        self.tau_alpha_p = 10.0
        self.tau_alpha_m = 1000.0
        self.alpha = np.zeros(n_neurons, dtype=float)
        self.theta = 2.0

        # synaptic variables
        self.tau_lr_0 = float(tau_lr0)
        self.tau_lr_i = np.full(n_neurons, float(tau_lr0), dtype=float)
        self.tau_decay = float(tau_decay)
        self.beta_w = 0.9

        # global inhibition
        self.I0 = float(I0)
        self.I1 = float(I1)
        self.I2 = float(I2)

        self.scaleE = 0
        self.scaleW = 0
        self.base = 0
        # tagged time (first t at which the FR is > theta for encoding and for recall)
        self.tag_t = np.zeros(n_neurons, dtype=float)
        self.tag_r = np.zeros(n_neurons, dtype=float)

        self.rec_w = np.zeros((n_neurons, n_neurons), dtype=float)
        self.rates = np.zeros(n_neurons, dtype=float)

    def step(self,ct,ctx,seq,seed = 0):
        """
        Perform one timestep of rate dynamics:
        input_FR: shape [n_neurons] or scalar
        """
        # total synaptic input
        input_FR = self.INPUT(ct,ctx,seq)
        # breakpoint()
        input_vector = input_FR + self.rec_w.dot(self.rates)
        self.rates = self.rates*(self.rates>1e-5)
        # global inhibition
        I_inhib = self.I0 + self.I1 * np.sum(self.rates) + self.I2 * np.sum(np.multiply(self.rates,self.rates))
        if ct == 0:
            np.random.seed(seed)
        e0 = np.zeros(self.n_neurons)
        e0 = np.random.normal(0.0, self.sigma, size=self.n_neurons)
        self.epsi_i0 = np.sqrt(e0*e0)
        # total input current
        input_current = input_vector - I_inhib

        # rate dynamics
        dr_dt = (-self.rates + np.maximum(0,self.epsi_i + input_current)) / self.tau

        # hebbian plasticity in recurrent weights
        hebbian_dw = (1. + self.scaleW * self.alpha) * (self.rates >= self.theta ) * self.rates.dot(self.rates.T)/self.tau_lr_i[:, None]  +  ((self.rates < self.theta)* self.rates.dot(self.rates.T)/self.tau_lr_i[:, None])
        hebbian_dw *= self.dt
        # gate_post = (self.rates >= self.theta).astype(float)[:, None]          # (N,1)
        # alpha_post = (1.0 + self.scaleW * self.alpha)[:, None]                 # (N,1)
        # hebb_core = np.outer(self.rates, self.rates)                           # (N,N)
        # base_lr = (1.0 / self.tau_lr_i)[:, None]                               # (N,1) postsyn learning rate

        # hebbian_dw = (gate_post * alpha_post + (1.0 - gate_post)) * base_lr * hebb_core * self.dt
        hebbian_dw =(1.0 / self.tau_lr_i)[:, None] * np.outer(self.rates, self.rates) * self.dt
        decay = (1.0 / self.tau_decay) * self.rec_w * self.dt

        # change in priming variable
        dalpha_dt = np.zeros(self.n_neurons)
        dalpha_dt += (self.tag_t !=0) * (ct >= 3000) * (ct<= 3100) * (self.tag_t < 3000) * (2 - self.alpha) / self.tau_alpha_p
        dalpha_dt += (self.tag_t !=0) * (ct >= 6000) * (ct<= 6100) * (self.tag_t > 3000)* (self.tag_t < 6000)*(2 - self.alpha) / self.tau_alpha_p
        dalpha_dt += (self.tag_t !=0) * (ct > 3100) * (self.tag_t < 3000) * (1 - self.alpha) / self.tau_alpha_m
        dalpha_dt += (self.tag_t !=0) * (ct > 6100) * (self.tag_t > 3000)* (self.tag_t < 6000) * (1 - self.alpha) / self.tau_alpha_m

        dexcdt = np.zeros(self.n_neurons)
        # dexcdt = ((self.rates > self.theta) * (self.scaleE * self.alpha + self.epsi_i0 - self.epsi_i ) / self.tau_e) + ((self.rates < self.theta) * (self.epsi_i0 - self.epsi_i) / self.tau_e)
        if dexcdt.max() != 0 and self.rates.max()>1e-5:
            # breakpoint()
            print("ex changed",dexcdt.max(),ct,self.rates.max())
        self.tag_t = self.tag_t + (self.rates > self.theta).astype(float) * (self.tag_t == 0).astype(float) * ct
        if ct > 25000:
            self.tag_r = self.tag_r + (self.rates > self.theta).astype(float) * (self.tag_r == 0).astype(float) * ct
        # if (self.rates > self.theta).all():
        #     breakpoint()
        self.rates += dr_dt * self.dt
        self.rec_w += (hebbian_dw - decay)
        np.clip(self.rec_w, 0.0, 1.0, out=self.rec_w)
        self.alpha += dalpha_dt*self.dt
        self.epsi_i += dexcdt*self.dt

        # clamp weights non-negative and <= 1
        return self.rates.copy(),input_FR

    def INPUT(self,t,ctx,seq,context_inp=10.):
        L = 0
        pol = 1
        for step in seq:
            L += np.tanh(t-step)*pol
            pol *= -1
        return (context_inp-self.base) * ctx * L/2 + self.base

# ---------- simulation ----------
start_seed = 0
np.random.seed(start_seed)

FR_history_all = []
EX_history_all = []
rec_weights_all = []
ff_weights_all = []
last_activity_all = []
input_history_all = []
alpha_history_all = []
n1 = 50
threshold = 2
FC_inp = 25
context_inp = 16 # scalar input (broadcast)
off_context_inp = 1.6
zero_input = np.zeros(n1, dtype=float)

ID = 1000
T = 32000
NUM_SIM = 1
N_off_days = 1
t_off = 100
IR = 100
Nrep = 10
t_Recall_start = 8000
t_recall = 100
IR_recall = 6000 - t_recall
N_recall = 4
# seqA = []
ctxA = np.zeros(n1)
ctxA[:int(n1/2)] = 1
# random.seed(0)
random.shuffle(ctxA)
#ctxB = np.zeros(N)
#ctxB[:int(overlap*N)] = ctxA[:int(overlap*N)]
#ctxB[int(overlap*N):] = 1 - ctxA[int(overlap*N):]
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
ov = np.arange(0,n1)
random.shuffle(ov)
overlap = 5/100
ctxB = 1 - ctxA
for n in ov[:int(overlap*n1)]:
    ctxB[n] = ctxA[n]
# breakpoint()
base = 0
for i in trange(NUM_SIM):
    current_t = 0
    base_E = np.random.normal(0.0, 1.0, size=n1)
    eps_all = np.sqrt(base_E*base_E)
    nn = twolayer_FF(n_neurons=n1, tau=20.0, dt=1.0,
                     act=relu, tau_lr0=2500, tau_decay=4000, I0=7, I1=0.5, I2=0.05)

    FR_history = []
    EX_history = []
    rec_weights = []
    input_history = []
    alpha_history = []
    high_threshold = 5

    # Initial drift
    for t1 in range(0, ID):
        # base_E = np.random.normal(0.0, 1.0, size=n)
        # eps_all = np.sqrt(base_E*base_E)
        # nn.epsi_i = eps_all
        next_FR,input_fr  = nn.step(current_t,ctxA,seqA)
        FR_history.append(next_FR.copy())
        input_history.append(input_fr.copy())
        current_t += 1
        EX_history.append(nn.epsi_i.copy())
        alpha_history.append(nn.alpha.copy())
    print("t =", current_t)
    seqA.append(current_t)
    # Off days with repetitions
    for day in range(N_off_days):
        day_activity = []
        for rep in range(Nrep):
            for t1 in range(0, t_off):
                # print(".", end="")
                # base_E = np.random.normal(0.0, 1.0, size=n)
                # eps_all = np.sqrt(base_E*base_E)
                # nn.epsi_i = eps_all
                next_FR,input_fr  = nn.step(current_t,ctxA,seqA)
                FR_history.append(next_FR.copy())
                input_history.append(input_fr.copy())
                current_t += 1
                EX_history.append(nn.epsi_i.copy())
                alpha_history.append(nn.alpha.copy())
            # day_activity.append(np.mean(FR_history[-t_off:], axis=0))
            print("t =", current_t)
            seqA.append(current_t)
            for t1 in range(0, IR):
                # base_E = np.random.normal(0.0, 1.0, size=n)
                # eps_all = np.sqrt(base_E*base_E)
                # nn.epsi_i = eps_all
                next_FR,input_fr  = nn.step(current_t,ctxA,seqA)
                FR_history.append(next_FR.copy())
                input_history.append(input_fr.copy())
                current_t += 1
                EX_history.append(nn.epsi_i.copy())
                alpha_history.append(nn.alpha.copy())
            print("t =", current_t)
            seqA.append(current_t)
        # rep_Activity.append(day_activity)
        rec_weights.append(nn.rec_w.copy())
        print("t =", current_t)
        seqA.append(current_t)
    # coast to recall start
    t_left = t_Recall_start - current_t
    for t1 in range(0, t_left):
        # base_E = np.random.normal(0.0, 1.0, size=n)
        # eps_all = np.sqrt(base_E*base_E)
        # nn.epsi_i = eps_all
        next_FR,input_fr  = nn.step(current_t,ctxA,seqA)
        FR_history.append(next_FR.copy())
        input_history.append(input_fr.copy())
        current_t += 1
        EX_history.append(nn.epsi_i.copy())
        alpha_history.append(nn.alpha.copy())
    print("t =", current_t)
    seqA.append(current_t)
    # recall blocks
    for rep in range(0, N_recall):
        for t1 in range(0, t_recall):
            # base_E = np.random.normal(0.0, 1.0, size=n)
            # eps_all = np.abs(base_E)
            # nn.epsi_i = eps_all
            next_FR,input_fr  = nn.step(current_t,ctxA,seqA)
            FR_history.append(next_FR.copy())
            input_history.append(input_fr.copy())
            current_t += 1
            EX_history.append(nn.epsi_i.copy())
            alpha_history.append(nn.alpha.copy())
        print("t =", current_t)
        seqA.append(current_t)
        for t1 in range(0, IR_recall):
            # base_E = np.random.normal(0.0, 1.0, size=n)
            # eps_all = np.abs(base_E)
            # nn.epsi_i = eps_all
            next_FR,input_fr  = nn.step(current_t,ctxA,seqA)
            FR_history.append(next_FR.copy())
            input_history.append(input_fr.copy())
            current_t += 1
            EX_history.append(nn.epsi_i.copy())
            alpha_history.append(nn.alpha.copy())
        rec_weights.append(nn.rec_w.copy())
        print("t =", current_t)
        seqA.append(current_t)

    # collect
    FR_history_all.append(FR_history)
    EX_history_all.append(EX_history)
    rec_weights_all.append(rec_weights)
    alpha_history_all.append(alpha_history)
    input_history_all.append(input_history)
print(seqA)
print("current_t == 32000", current_t == T)
FR_history_all = np.stack(FR_history_all)
EX_history_all = np.stack(EX_history_all)
alpha_history_all = np.stack(alpha_history_all)
input_history_all = np.stack(input_history_all)
breakpoint()

inp_data_folder = "./data/epigenetic_paper"
op_plot_folder = "./plots/epigenetic_paper"
os.makedirs(inp_data_folder, exist_ok=True)
os.makedirs(op_plot_folder, exist_ok=True)
np.save(f"{inp_data_folder}/FR_history.npy", FR_history_all)
np.save(f"{inp_data_folder}/EX_history.npy", EX_history_all)
np.save(f"{inp_data_folder}/alpha_history.npy", alpha_history_all)
np.save(f"{inp_data_folder}/rec_W.npy", rec_weights_all)
np.save(f"{inp_data_folder}/input_history.npy", input_history_all)

sim_params = {
    "n": n,
    "threshold": threshold,
    "ID": ID,
    "N_off_days": N_off_days,
    "t_off": t_off,
    "IR": IR,
    "Nrep": Nrep,
    "start_seed": start_seed
}
data = {
    "model_params": {
        "n_neurons": nn.n_neurons,
        "tau": nn.tau,
        "dt": nn.dt,
        "Tau_lr": nn.tau_lr_0,
        "tau_decay_r": nn.tau_decay,
        "I0": nn.I0,
        "I1": nn.I1,
        "I2": nn.I2,
    },
    "simulation_params": sim_params
}
# filename = f"{inp_data_folder}/all_params.json"
# with open(filename, "w") as f:
#     json.dump(data, f, indent=4)

FR_history_th = (FR_history_all >= threshold).astype(float) * FR_history_all
tn, un = get_tagged_neurons(FR_history_all[0].T, threshold)
t_points = np.arange(1, current_t + 1, 1) * 1e-3
for t1 in tn:
    plt.plot(t_points, FR_history_all[0][:, t1], c='r')
for t1 in un:
    plt.plot(t_points, FR_history_all[0][:, t1], c='k')
plt.hlines(y=2, xmin=0, xmax=t_points[-1], colors='k', linestyles='--')
plt.show()

# breakpoint()  # optional

plot_activity_n_excitability_time(
    [FR_history_th[0].T, EX_history_all[0].T],
    titles=['Neuronal Activity', "Neuronal Excitability"],
    fname=f"{op_plot_folder}/Activity_n_excitability.svg",
    cmaps=['binary', 'Blues']
)

plot_activity_n_excitability_time(
    [input_history_all.T, alpha_history_all[0].T],
    titles=['Input', "Alpha"],
    fname=f"{op_plot_folder}/input_and_alpha.svg",
    cmaps=['binary', 'Blues']
)

labs = ["Encoding"] + [f"Recall {i+1}" for i in range(N_recall)]
plot_weights_over_time(
    rec_weights_all[0],
    titles=labs,
    fname=f"{op_plot_folder}/Rec_w.svg",
    cmaps='gray_r'
)
