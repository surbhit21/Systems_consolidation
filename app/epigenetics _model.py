import matplotlib.pyplot as plt
import json
import numpy as np
from plotting_widget import *
import torch 
import torch.nn.functional as F
from Utilities import average_firing_rates_with_active,ensamble_overlap,get_tagged_neurons
from tqdm import trange
start_seed = 120
torch.manual_seed(start_seed)
class twolayer_FF:
    def __init__(self, n_neurons,epsi_i0,tau=20.0, dt=1.0, act=torch.relu, tau_lr0=800, tau_decay=1000, I0=1, I1=0.05, I2=0.001):
        # netowrk and simulation propertites
        self.n_neurons = n_neurons
        self.dt = dt  # discretization step

        # neuronal properties
        self.tau = tau  # rate constant 
        self.act = act  # activation function
        # excitability params
        self.epsi_i0 = epsi_i0
        self.epsi_i = epsi_i0
        self.beta_e = 0.9

        # epigenetic priming variable
        self.tau_alpha_p = 10
        self.tau_alpha_m = 1000
        self.alpha = torch.zeros(n_neurons)
        self.theta = 2

 
        # synaptic variables
        self.tau_lr_0 = tau_lr0
        self.tau_lr_i = tau_lr0*torch.ones(n_neurons)  # learning rate for synaptic weights
        self.tau_decay = tau_decay # decay rate for synaptic weights
        self.beta_w = 0.9
       
        # global inhibition
        self.I0 = I0
        self.I1 = I1
        self.I2 = I2
        

        self.rec_w = torch.zeros(n_neurons, n_neurons)
        self.rates = torch.zeros(n_neurons)

    def step(self, input_FR,t):
        """
        Perform one timestep of rate dynamics:
        input_vector: shape [n_neurons]
        """
        # calculating the input to the RNN
        input_vector = input_FR + self.rec_w @ self.rates
        
        # breakpoint()
        # print(input_vector.max())
        # blanket inhibition to the RNN
        I_inhib = self.I0 + self.I1 * torch.sum(self.rates) + self.I2 * torch.sum(torch.pow(self.rates,2))
        
        # print(input_vector.max(),I_inhib)
        # total input to the RNN
        input_current =  input_vector -  I_inhib 
        
        # rate change as the nonlinear ODE
        dr_dt = (-self.rates +   self.act(self.epsi_i + input_current )) / self.tau
        self.rates += (dr_dt * self.dt)
        
        # hebbian plasticity in input weights
        hebbian_dw = (1/self.tau_lr_i) * torch.outer(self.rates, self.rates) * self.dt
        decay = (1./self.tau_decay) * self.rec_w * self.dt
        self.rec_w += (hebbian_dw - decay)
        # print(input_vector.max(),I_inhib,dr_dt.max(),hebbian_dw.max(),decay.max())
       
        self.rec_w = torch.clamp(self.rec_w, 0.0, 1.0)  # Ensure weights are non-negative
        if dr_dt.max() > 0.02:
            # breakpoint()
            print(dr_dt.max())
        return self.rates.detach().clone()
    
    def lr_dynamics(self):
        for xdx, rate in enumerate(self.rates):
            if rate > self.theta:
                self.tau_lr_i[xdx] = self.tau_lr_0 / (1 + self.beta_w * self.alpha[xdx])

    def alpha_dynamics(self,i,up_steady_or_down):
        # primed_neurons = (self.rates > self.theta).astype(float)
        if up_steady_or_down == 1:
            dalpha_i_dt = (2-self.alpha[i])/self.tau_alpha_p
        elif up_steady_or_down == 2:
            dalpha_i_dt = 0
        else:
            dalpha_i_dt = (1-self.alpha[i])/self.tau_alpha_m
        self.alpha[i] += (dalpha_i_dt)*self.dt

    def epsi_dynamics(self):
        # primed_neurons = (self.rates > self.theta).astype(float)
        for xdx, rate in enumerate(self.rates):
            if rate > self.theta:
                self.epsi[xdx] = self.epsi_i0 * (1 + self.beta_e * self.alpha[xdx])
    

 
 
FR_history = []
EX_history = []
rec_weights = []
ff_weights = []
last_activity = []
input_history = []
#
n = 50
threshold = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
# base_E[:off_set] += 2
FC_inp = 25
context_inp = 16 #*torch.ones(n)
off_context_inp = 1.6 
zero_input = torch.zeros(n)
g = torch.Generator(device=device).manual_seed(2024)
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
FR_history_all = []
EX_history_all = []
rec_weights_all = []
ff_weights_all = []
last_activity_all = []
input_history_all = []
chi2 = torch.distributions.Chi2(df=torch.tensor(1.0, device=device))
# eps_all = chi2.sample((T,n)).to(device)  # shape: [T, N]
# print(eps_all.max(),eps_all.min())

for i in trange(NUM_SIM):
    current_t = 0
    # torch.manual_seed(start_seed)
    base_E =torch.normal(0,1,size=(n,))
    eps_all = torch.sqrt(base_E*base_E)
    nn = twolayer_FF(n_neurons=n, tau=20.0,epsi_i0=eps_all, dt=1, act=torch.relu, tau_lr0=2500, tau_decay=4000, I0=7, I1=0.5, I2=0.05)
    FR_history = []
    EX_history = []
    rec_weights = []
    ff_weights = []
    last_activity = []
    input_history = []
    rep_Activity = []
    high_threshold = 5 
    for t1 in range(0,ID):
        base_E =torch.normal(0,1,size=(n,))
        eps_all = torch.sqrt(base_E*base_E)
        nn.epsi_i = eps_all
        next_FR = nn.step(zero_input,current_t)
        FR_history.append(next_FR.detach().clone().numpy())
        current_t += 1
        EX_history.append(nn.epsi_i.detach().clone().numpy())
    # breakpoint()
    for day in range(N_off_days):
        day_activity = []
        # torch.manual_seed(120+day)
        # inp_to_network = context_inp
        for rep in range(Nrep):
            for t1 in range(0,t_off):
                print(".",end="")
                base_E =torch.normal(0,1,size=(n,))
                eps_all = torch.sqrt(base_E*base_E)
                nn.epsi_i = eps_all
                next_FR = nn.step(context_inp,current_t)
                FR_history.append(next_FR.detach().clone().numpy())
                current_t += 1
                EX_history.append(nn.epsi_i.detach().clone().numpy())
                # input_history.append(context_inp.numpy())
            day_activity.append(np.mean(FR_history[-t_off:],axis=0))    
            # current_t += IR
            for t1 in range(0,IR):
                base_E =torch.normal(0,1,size=(n,))
                eps_all = torch.sqrt(base_E*base_E)
                nn.epsi_i = eps_all
                next_FR = nn.step(zero_input,current_t)
                FR_history.append(next_FR.detach().clone().numpy())
                current_t += 1
                EX_history.append(nn.epsi_i.detach().clone().numpy())
                # EX_history.append(nn.excitability.detach().clone().numpy())
                # input_history.append(zero_input.numpy())
        rep_Activity.append(day_activity)
        rec_weights.append(nn.rec_w.detach().clone().numpy())
    t_left = t_Recall_start - current_t
    for t1 in range(0,t_left):
        base_E =torch.normal(0,1,size=(n,))
        eps_all = torch.sqrt(base_E*base_E)
        nn.epsi_i = eps_all
        next_FR = nn.step(zero_input,current_t)
        FR_history.append(next_FR.detach().clone().numpy())
        current_t += 1
        EX_history.append(nn.epsi_i.detach().clone().numpy())
        # EX_history.append(nn.excitability.detach().clone().numpy())
    
    for rep in range(0,N_recall):
        # current_t += t_recall
        for t1 in range(0,t_recall):
            base_E =torch.normal(0,1,size=(n,))
            eps_all = torch.sqrt(base_E*base_E)
            nn.epsi_i = eps_all
            next_FR = nn.step(context_inp,current_t)
            FR_history.append(next_FR.detach().clone().numpy())
            current_t += 1
            EX_history.append(nn.epsi_i.detach().clone().numpy())
            # EX_history.append(nn.excitability.detach().clone().numpy())
            # input_history.append(context_inp.numpy())
        # day_activity.append(np.mean(FR_history[-t_off:],axis=0))    
        # current_t += IR_recall
        for t1 in range(0,IR_recall):
            base_E =torch.normal(0,1,size=(n,))
            eps_all = torch.sqrt(base_E*base_E)
            nn.epsi_i = eps_all
            next_FR = nn.step(zero_input,current_t)
            FR_history.append(next_FR.detach().clone().numpy())
            current_t += 1
            EX_history.append(nn.epsi_i.detach().clone().numpy())
            # EX_history.append(nn.excitability.detach().clone().numpy())
            # input_history.append(zero_input.numpy())
        rec_weights.append(nn.rec_w.detach().clone().numpy())
    # recall phase
    FR_history_all.append(FR_history)
    EX_history_all.append(EX_history)
    rec_weights_all.append(rec_weights)
    # ff_weights_all.append(ff_weights)
    # last_activity_all.append(lastsactivity)
    # input_history_all.append(input_history)


print("current_t == 32000",current_t == T)
FR_history_all = np.stack(FR_history_all)
# input_history = np.stack(input_history)
EX_history_all = np.stack(EX_history_all)
# last_activity_all = np.stack(last_activity_all)
inp_data_folder = "./data/epigenetic_paper"
op_plot_folder = "./plots/epigenetic_paper"
os.makedirs(inp_data_folder, exist_ok=True)
os.makedirs(op_plot_folder, exist_ok=True)
np.save("{}/FR_history.npy".format(inp_data_folder),FR_history_all)
np.save("{}/EX_history.npy".format(inp_data_folder),EX_history_all)
np.save("{}/rec_W.npy".format(inp_data_folder),rec_weights_all)
# np.save("./data/epigenetic_paper/last_activity.npy",last_activity_all)
# np.save("./data/Reimagined3/input_history.npy",input_history_all)  
sim_params = {
    "n": n,
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
            "Tau_lr": nn.tau_lr_0,
            "tau_decay_r": nn.tau_decay,
            "I0": nn.I0,
            "I1": nn.I1,
            "I2": nn.I2,
            # "excitability": model.excitability.detach().cpu().numpy().tolist(),
        },
        "simulation_params": sim_params
    }
filename = "{}/all_params.json".format(inp_data_folder)

with open(filename, "w") as f:
        json.dump(data, f, indent=4)
FR_history_th = (FR_history_all >= threshold).astype(float)*FR_history_all
tn,un = get_tagged_neurons(FR_history_all[0].T,threshold)
t_points = np.arange(1,current_t+1,1)*1e-3
for t1 in tn:
    plt.plot(t_points,FR_history_all[0][:,t1],c='r')
for t1 in un:
    plt.plot(t_points,FR_history_all[0][:,t1],c='k')
plt.hlines(xmin = 0,xmax=t_points[-1],y=2,colors='k', linestyles='--')
plt.show()
breakpoint()
plot_activity_n_excitability_time([FR_history_th[0].T,EX_history_all[0].T],
                       titles=['Neuronal Activity',
                                "Neuronal Excitability"],
                       fname="{}/Activity_n_excitability.svg".format(op_plot_folder),
                       cmaps=['binary', 'Blues'])

labs = ["Encoding"] + [f"Recall {i+1}" for i in range(N_recall)]
plot_weights_over_time(rec_weights_all[0],
                       titles=  labs,
                       fname="{}/Rec_w.svg".format(op_plot_folder),
                       cmaps='gray_r')