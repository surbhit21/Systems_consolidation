import matplotlib.pyplot as plt
import json
from LoadNPLot2 import PlotAll
import numpy as np
from plotting_widget import *
import torch 
import torch.nn.functional as F
from Utilities import average_firing_rates_with_active,ensamble_overlap
from tqdm import trange
start_seed = 5
torch.manual_seed(start_seed)
class twolayer_FF:
    def __init__(self, n_inp, n_MTL,n_CTX,baseline_e,base_e_ctx,tau=20.0, dt=1.0, act=F.softplus, 
                 lr=1/800, decay_r=1/1000, 
                 lr_ctx = 1/1200, decay_r_ctx = 1e-5,
                 lr_mtl_ctx = 1/1000,decay_mtl_ctx = 1e-5, 
                 lr_ctx_mtl = 1/1000, decay_ctx_mtl = 1e-5,
                 threshold=5,   
                 lr_op = 1/1000,
                 I0=1, I1=0.05, I2=0.001,Iw = 0.01,
                 I0_ctx=1, I1_ctx=0.05, I2_ctx=0.001,Iw_ctx = 0.01, tag_threshold=3.):
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
        self.lr_mtl_ctx = lr_mtl_ctx
        self.lr_ctx_mtl = lr_ctx_mtl
        self.decay_mtl_ctx = decay_mtl_ctx
        self.decay_ctx_mtl = decay_ctx_mtl
        self.lr_op = lr_op
        self.I0 = I0
        self.I1 = I1
        self.I2 = I2
        self.Iw = Iw
        self.I0_ctx = I0_ctx
        self.I1_ctx = I1_ctx
        self.I2_ctx = I2_ctx
        self.Iw_ctx = Iw_ctx
        self.excitability = baseline_e
        self.excitability_ctx = base_e_ctx
        self.plas_threshold = 0.2
        self.act_threshold = 0#torch.abs(torch.normal(0,0.7,size=(n_MTL,)))
        self.act_threshold_ctx = 0# torch.abs(torch.normal(0,0.7,size=(n_ctx,)))
        self.threshold = threshold
        self.tag_threshold = tag_threshold
        self.FF_MTL_CTX = (1./np.sqrt(n_MTL)) * torch.abs(torch.normal(0,0.05,size=(n_CTX,n_MTL)))
        self.FB_CTX_MTL = 0 * (1./np.sqrt(n_MTL)) * torch.abs(torch.normal(0,0.05,size=(n_MTL,n_CTX)))
        self.FF_plas = 1.0
        self.gain_FF = 1.0
        self.gain_ctx = 1.
        self.gain_hpc = 1.
        self.op_neuron = 1
        self.on = 1.0  # plasticity on/off
        self.on_ctx = 1.0
        self.tagged_ACC = np.zeros(n_CTX)
        self.tagged_HPC = np.zeros(n_MTL)
        # Initialize random input weights
        self.input_w = torch.abs(torch.normal(0,0.05,size=(n_inp,n_MTL)))
        # self.input_w = torch.clamp(self.input_w, 0.0, 0.1)
        self.rec_w = torch.zeros(n_MTL, n_MTL)
        self.rec_w_ctx = torch.zeros(self.n_CTX, self.n_CTX)
        
        self.mtl_op_w = 0*torch.abs(torch.normal(0,0.05,size=(self.op_neuron,n_MTL)))
        self.ctx_op_w = 0*torch.abs(torch.normal(0,0.05,size=(self.op_neuron,n_CTX)))
        # Zero initial rate state
        self.rates_ctx = torch.zeros(self.n_CTX)
        self.rates = torch.zeros(n_MTL)
        self.op_rate = torch.zeros(self.op_neuron)
        # self.

    def step(self, input_FR,input_FR_ctx,op_FR, op_signal = 0,day_c=0,phase=None,ct=1):
        """
        Perform one timestep of rate dynamics:
        input_vector: shape [n_MTL]
        """
        # # calculating the input to the RNN
        # self.rates *= (self.rates > 1e-5).float()  # Ensure rates are non-negative
        # self.rates_ctx *= (self.rates_ctx > 1e-5).float()

        input_vector = input_FR + self.rec_w @ self.rates - self.FB_CTX_MTL @ self.rates_ctx
        input_CTX = input_FR_ctx + self.rec_w_ctx @ self.rates_ctx + (self.gain_FF * self.FF_MTL_CTX @self.rates)
        input_op = op_FR + self.mtl_op_w @ self.rates + self.ctx_op_w @ self.rates_ctx

        # breakpoint()
        # print(input_vector.max(),input_CTX.max(),self.rec_w.max(),self.rec_w_ctx.max())
        # blanket inhibition to the RNN
  

        I_inhib = self.I0 + self.I1 * torch.sum(self.rates) + self.I2 * torch.sum(self.rates**2)
        I_inhib_ctx = self.I0_ctx + self.I1_ctx * torch.sum(self.rates_ctx) + self.I2_ctx * torch.sum(self.rates_ctx**2)

        # neurons that are above threshold are tagged as part of the engram
        self.tagged_HPC = self.tagged_HPC + (((self.rates > self.tag_threshold).detach().clone().numpy().astype(float) * (self.tagged_HPC == 0).astype(float)) * day_c)
        self.tagged_ACC = self.tagged_ACC + (((self.rates_ctx > self.tag_threshold).detach().clone().numpy().astype(float) * (self.tagged_ACC == 0).astype(float)) * day_c)


        # print(I_inhib)
        # total input to the RNN
        input_current =  input_vector -  I_inhib 
        input_ctx = input_CTX - I_inhib_ctx

        if phase == "Encoding" and ct%500 == 0:
            # breakpoint()
            print(input_current.max(),input_ctx.max(),self.rates.max(),self.rates_ctx.max(),I_inhib,I_inhib_ctx,input_vector.max(),input_CTX.max(),input_op,phase,ct)
        if phase == "NREM" and ct%200 == 0:
            print(input_current.max(),input_ctx.max(),self.rates.max(),self.rates_ctx.max(),I_inhib,I_inhib_ctx,input_vector.max(),input_CTX.max(),input_op,phase,ct)
        if phase == "REM" and ct%500 == 0:
            # breakpoint()
            print(input_current.max(),input_ctx.max(),self.rates.max(),self.rates_ctx.max(),I_inhib,I_inhib_ctx,input_vector.max(),input_CTX.max(),input_op,phase,ct)
        # rate change as the nonlinear ODE
        dr_dt = (-self.rates +   self.act(self.gain_hpc * (self.excitability + input_current + self.act_threshold))) / self.tau
        dr_ctx_dt = (-self.rates_ctx +   self.act( self.gain_ctx * (self.excitability_ctx + input_ctx  + self.act_threshold_ctx))) / self.tau
        dr_op_dt = (-self.op_rate + self.act(input_op)) / self.tau


        
        self.rates += (dr_dt * self.dt)
        self.rates_ctx += (dr_ctx_dt * self.dt)
        self.op_rate += (dr_op_dt*self.dt)

        # self.rates = torch.clamp(self.rates, 0.0, 10.)  # Ensure rates are non-negative
        # self.rates_ctx = torch.clamp(self.rates_ctx, 0.0, 10.)
        
        post_mask = self.rates > self.plas_threshold
        # hebbian plasticity in recurrent weights
        heb_dw = self.on * self.lr * torch.outer(self.rates*post_mask, self.rates*post_mask) * self.dt
        decay = self.decay_r * self.rec_w * self.dt
        self.rec_w += (heb_dw - decay)
        # plasticity in CTX -> MTL weights
        hebb_dw_ctx_mtl = self.on * self.lr_ctx_mtl * torch.outer(self.rates*post_mask, self.rates_ctx) * self.dt
        decay_ctx_mtl = self.decay_ctx_mtl * self.FB_CTX_MTL * self.dt
        self.FB_CTX_MTL += (hebb_dw_ctx_mtl - decay_ctx_mtl)
        # plasticity in MTL -> output weights
        hebb_mtl_op = self.lr_op * op_signal * torch.outer(self.op_rate, self.rates*post_mask) * self.dt
        decay_mtl_op = 1e-4 * self.mtl_op_w * self.dt
        self.mtl_op_w += (hebb_mtl_op - decay_mtl_op)


        # plasticity in recurrent weights in CTX
        post_mask = self.rates_ctx > self.plas_threshold
        hebb_dw_ctx = self.on_ctx * self.lr_ctx * torch.outer(self.rates_ctx*post_mask, self.rates_ctx*post_mask) * self.dt
        decay_ctx = self.decay_r_ctx * self.rec_w_ctx * self.dt
        self.rec_w_ctx += (hebb_dw_ctx - decay_ctx)
        # plasticity in MTL -> CTX weights
        hebb_dw = self.FF_plas * self.on_ctx * self.lr_mtl_ctx * torch.outer(self.rates_ctx*post_mask, self.rates*post_mask) * self.dt
        decay =  self.FF_plas * self.decay_mtl_ctx * self.FF_MTL_CTX * self.dt
        self.FF_MTL_CTX += (hebb_dw - decay)

        hebb_ctx_op = self.lr_op * op_signal * torch.outer(self.op_rate, self.rates_ctx*post_mask) * self.dt
        decay_ctx_op = 1e-4 * self.ctx_op_w * self.dt
        self.ctx_op_w += (hebb_ctx_op - decay_ctx_op)

        sum_w = torch.sum(self.ctx_op_w)
        if sum_w > 0:
            self.ctx_op_w /= sum_w
        self.ctx_op_w = torch.clamp(self.ctx_op_w,min=0)
        sum_w = torch.sum(self.mtl_op_w)
        if sum_w > 0:
            self.mtl_op_w /= sum_w
        self.mtl_op_w = torch.clamp(self.mtl_op_w,min=0)

        # self.rates = torch.clamp(self.rates, 0.0, 15)  # Ensure rates are non-negative
        self.rec_w = torch.clamp(self.rec_w, 0.0, 1.0)  # Ensure weights are non-negative
        self.rec_w_ctx = torch.clamp(self.rec_w_ctx, 0.0, 1.0)
        self.FF_MTL_CTX = torch.clamp(self.FF_MTL_CTX, 0.0, 1.)  # Ensure weights are non-negative
        self.FB_CTX_MTL = torch.clamp(self.FB_CTX_MTL, 0.0, 1.)
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
N_neurons_per_day = 20
n =  10 + (N_off_days)* N_neurons_per_day#10 default + 20 neurons per off day
n_inp = n 
n_ctx = n 
E_fl = 2.2
E_fl_ctx = 2.2
E_mod = 2.0
max_e = 10
E_ref = 0.7
threshold = 5.0
off_set = 0



# base_E[:off_set] += 2
IP_plasticity_limit = 1
if E_mod == 0.0:
    sim_name = "CNT_fast_drift_wo_IP_lowI"
else:
    sim_name = "CNT_fast_drift_with_limited{}_IP_lowI".format(IP_plasticity_limit)
notes = "2 region model with slow drift due to low excitability boosts in both regions with intrinsic plasticity. ACC neurons that are part of the FC engram get an extra boost in excitability during off days."
FC_inp = 15
FC_d0_inp = 15
FC_inp_ctx = 15
input = FC_inp*torch.ones(n_inp)
zero_input = torch.zeros(n_inp)
mu_ex = 0
sigma_ex = 1.0

ID = 1000
dt = 1
NUM_SIM = 5
t_off = 100
IR = 100
Nrep = 10
total_time = 0
input_history_all = []

FR_history_all = []
FR_ctx_history_all = []
FR_op_history_all = []
last_activity_all = []
last_activity_ctx_all = []

EX_history_all = []
EX_history_ctx_all = []

rec_weights_all = []
rec_ctx_weights_all = []

mtl_ctx_weights_all = []
ctx_mtl_weights_all = []

ctx_op_weights_all = []
mtl_op_weights_all =[]


dob = []
off_days = [0,1,2,3,7,10]#14,21,28,29]

# t_encoding = 1000


for i in trange(NUM_SIM):
    t_series = []
    total_time = 0
    torch.manual_seed(start_seed + i)
    base_E = torch.abs(torch.normal(mu_ex,sigma_ex,size=(n,)))
    base_e_ctx = torch.abs(torch.normal(mu_ex,sigma_ex,size=(n_ctx,)))
    nn = twolayer_FF(n_inp=n_inp, n_MTL=n,n_CTX=n_ctx, baseline_e = base_E.clone(),
                     base_e_ctx=base_e_ctx.clone(), tau=20.0, dt=dt, act=torch.relu,
                    lr=1./3000, decay_r=1./3500, 
                    lr_ctx = 1e-6,decay_r_ctx=0.,
                    lr_mtl_ctx = 1/3000., decay_mtl_ctx= 1e-6,
                    lr_ctx_mtl =0., decay_ctx_mtl=0.,
                    threshold=threshold,
                    lr_op = 1e-3,
                    I0=5, I1=0.7, I2=0.04,
                    I0_ctx=5., I1_ctx=0.6, I2_ctx=0.04)
    input_history = []

    FR_history = []
    FR_history_ctx = []
    FR_op_history = []
    last_activity = []
    last_activity_ctx = []

    EX_history = []
    EX_history_ctx = []

    rec_weights = []
    rec_ctx_weights = []
    mtl_ctx_weights = []
    ctx_mtl_weights = []

    rep_Activity = []
    high_threshold = 5 

    mtl_op_weights = []
    ctx_op_weights = []
    nn.excitability = base_E.clone()
    day = 0

    nn.excitability[off_set+(day)*N_neurons_per_day:off_set+(day)*N_neurons_per_day+N_neurons_per_day] += E_fl
    nn.excitability_ctx  = base_e_ctx.clone()
    nn.excitability_ctx[(off_set+(day)*N_neurons_per_day): (off_set+(day)*N_neurons_per_day+N_neurons_per_day)] += E_fl_ctx

    for t in range(ID):
        next_FR,FR_ctx,FR_op = nn.step(zero_input,zero_input,0,0,day_c=-1,phase="Burnoff",ct=total_time+t)
        FR_history.append(next_FR)
        FR_history_ctx.append(FR_ctx)
        FR_op_history.append(FR_op)
        EX_history.append(nn.excitability.detach().clone().numpy())
        EX_history_ctx.append(nn.excitability_ctx.detach().clone().numpy())
        input_history.append(zero_input.numpy())
    total_time += ID
    t_series.append(total_time)
    for day in range(N_off_days):
        day_activity = []
        day_activity_ctx = []
        nn.excitability = base_E.clone()
        nn.excitability[off_set+(day)*N_neurons_per_day:off_set+(day)*N_neurons_per_day+N_neurons_per_day] += E_fl
        nn.excitability_ctx  = base_e_ctx.clone()
        new_neurons = range(off_set + day*N_neurons_per_day, off_set + day*N_neurons_per_day + N_neurons_per_day, 1)
        if day <= IP_plasticity_limit:
            FC_ctx_active_neurons = np.where(nn.tagged_ACC == 1.)[0]
            FC_HPC_active_neurons = np.where(nn.tagged_HPC == 1.)[0]
            # breakpoint()
            # print("persistent neurons: CTX = {}, MTL = {}".format(FC_ctx_active_neurons, FC_HPC_active_neurons), len(FC_ctx_active_neurons), len(FC_HPC_active_neurons))
            nn.excitability_ctx[FC_ctx_active_neurons] += (E_mod) 
            for n1 in new_neurons:
                if not n1 in FC_ctx_active_neurons:
                    nn.excitability_ctx[n1] += E_fl_ctx
        else:
            nn.excitability_ctx[new_neurons] += E_fl_ctx
        # nn.excitability_ctx[new_neurons] += E_fl_ctx
        # inp_to_network = input      
        ctx_inp =  FC_inp_ctx
        if day == 0:
            op_inp  = 5.
            op_learning = 1.
            phase = "Encoding"
            input = FC_d0_inp*torch.ones(n_inp)
            ctx_inp = input
        else:
            op_inp = 0.
            op_learning = 0.
            input = FC_inp*torch.ones(n_inp)
            ctx_inp = input
            phase = "NREM"
        for rep in range(Nrep):
            if day in dob:
                # nn.gain_hpc = 0.0
                # nn.TurnOFF_FB()
                nn.on_ctx = 0.0
            else:
                # nn.gain_hpc  = 1.0
                # nn.TurnON_FB()
                nn.on_ctx = 1.0
            
            nn.FF_plas = 1.0
            nn.gain_FF = 1.0
            for t in range(t_off):    
                next_FR,FR_ctx,FR_op = nn.step(input,ctx_inp,op_inp,op_learning,day_c=day+1,phase=phase,ct=total_time+t)
                FR_history.append(next_FR)
                FR_history_ctx.append(FR_ctx)
                FR_op_history.append(FR_op)
                EX_history.append(nn.excitability.detach().clone().numpy())
                EX_history_ctx.append(nn.excitability_ctx.detach().clone().numpy())
                input_history.append(input.numpy())
            day_activity.append(np.mean(FR_history[-t_off//2:],axis=0))
            day_activity_ctx.append(np.mean(FR_history_ctx[-t_off//2:],axis=0))
            # breakpoint()
            total_time += t_off
            t_series.append(total_time)                
            for t in range(IR):
                next_FR,FR_ctx, FR_op = nn.step(zero_input,zero_input,0,0,day_c=day+1,phase="IR",ct=total_time)
                FR_history.append(next_FR)
                FR_history_ctx.append(FR_ctx)
                FR_op_history.append(FR_op)
                EX_history.append(nn.excitability.detach().clone().numpy())
                EX_history_ctx.append(nn.excitability_ctx.detach().clone().numpy())
                input_history.append(zero_input.numpy())
            total_time += IR
            if rep != Nrep -1:
                t_series.append(total_time)
        # breakpoint()
        # if day in off_days:
        nn.FF_plas = 0.0
        nn.gain_FF = 0.0
        rec_weights.append(nn.rec_w.detach().clone().numpy())
        rec_ctx_weights.append(nn.rec_w_ctx.detach().clone().numpy())
        mtl_ctx_weights.append(nn.FF_MTL_CTX.detach().clone().numpy())
        ctx_mtl_weights.append(nn.FB_CTX_MTL.detach().clone().numpy())
        mtl_op_weights.append(nn.mtl_op_w.detach().clone().numpy())
        ctx_op_weights.append(nn.ctx_op_w.detach().clone().numpy())
            
        rep_Activity.append(day_activity)
        last_activity.append(day_activity[-1])
        last_activity_ctx.append(day_activity_ctx[-1])
        for t in range(ID):
            next_FR,FR_ctx,FR_op = nn.step(zero_input,zero_input,0,day_c=day+1,phase="REM",ct=total_time+t)
            FR_history.append(next_FR)
            FR_history_ctx.append(FR_ctx)
            FR_op_history.append(FR_op)
            EX_history.append(nn.excitability.detach().clone().numpy())
            EX_history_ctx.append(nn.excitability_ctx.detach().clone().numpy())
            input_history.append(zero_input.numpy())
        total_time += ID
        t_series.append(total_time)
    # breakpoint()
    input_history_all.append(input_history)
    FR_history_all.append(FR_history)
    FR_ctx_history_all.append(FR_history_ctx)
    FR_op_history_all.append(FR_op_history)
    last_activity_all.append(last_activity)
    last_activity_ctx_all.append(last_activity_ctx)
    EX_history_all.append(EX_history)
    EX_history_ctx_all.append(EX_history_ctx)
    rec_weights_all.append(rec_weights)
    rec_ctx_weights_all.append(rec_ctx_weights)
    mtl_ctx_weights_all.append(mtl_ctx_weights)
    ctx_mtl_weights_all.append(ctx_mtl_weights)
    mtl_op_weights_all.append(mtl_op_weights)
    ctx_op_weights_all.append(ctx_op_weights)
FR_history_all = np.stack(FR_history_all)
FR_ctx_history_all = np.stack(FR_ctx_history_all)
FR_op_history_all  = np.stack(FR_op_history_all)


input_history = np.stack(input_history)
EX_history_all = np.stack(EX_history_all)
EX_history_ctx_all = np.stack(EX_history_ctx_all)
last_activity_all = np.stack(last_activity_all)
last_activity_ctx_all = np.stack(last_activity_ctx_all)
rec_weights_all = np.stack(rec_weights_all)
rec_ctx_weights_all = np.stack(rec_ctx_weights_all)
mtl_ctx_weights_all = np.stack(mtl_ctx_weights_all)
ctx_mtl_weights_all = np.stack(ctx_mtl_weights_all)
mtl_op_weights_all = np.stack(mtl_op_weights_all)
ctx_op_weights_all = np.stack(ctx_op_weights_all)
op_data_folder = "./data/{}".format(sim_name)
op_plot_folder = "./plots/{}".format(sim_name)# --- IGNORE ---

breakpoint()
os.makedirs(op_data_folder, exist_ok=True)
np.save("{}/FR_history.npy".format(op_data_folder),FR_history_all)
np.save("{}/FR_history_ctx.npy".format(op_data_folder),FR_ctx_history_all)
np.save("{}/FR_history_op.npy".format(op_data_folder),FR_op_history_all)
np.save("{}/EX_history.npy".format(op_data_folder),EX_history_all)
np.save("{}/EX_history_ctx.npy".format(op_data_folder),EX_history_ctx_all)
np.save("{}/last_activity.npy".format(op_data_folder),last_activity_all)
np.save("{}/last_activity_ctx.npy".format(op_data_folder),last_activity_ctx_all)
np.save("{}/input_history.npy".format(op_data_folder),input_history_all)
np.save("{}/rec_weights.npy".format(op_data_folder),rec_weights_all)
np.save("{}/mtl_op_weights.npy".format(op_data_folder),mtl_op_weights_all)
np.save("{}/ctx_op_weights.npy".format(op_data_folder),ctx_op_weights_all)
np.save("{}/rec_ctx_weights.npy".format(op_data_folder),rec_ctx_weights_all)
np.save("{}/mtl_ctx_weights.npy".format(op_data_folder),mtl_ctx_weights_all)
np.save("{}/ctx_mtl_weights.npy".format(op_data_folder),ctx_mtl_weights_all)

# np.save("./data/Reimagined3/input_history.npy",input_history_all)  

sim_params = {
    "n": n,
    "n_inp": n_inp,
    "n_ctx": n_ctx,
    "FC_inp":FC_inp,
    "E_fl": E_fl,
    "E_fl_ctx":E_fl_ctx,
    "E_mod": E_mod,
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
    "notes": notes,
    "off_days": off_days,
    "t_series": t_series,
    "IP_plasticity_limit": IP_plasticity_limit
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
            "lr_mtl_ctx": nn.lr_mtl_ctx,
            "decay_mtl_ctx": nn.decay_mtl_ctx,
            "lr_ctx_mtl": nn.lr_ctx_mtl,
            "decay_ctx_mtl": nn.decay_ctx_mtl,
            # "plas_threshold": nn.plas_threshold,
            "I0": nn.I0,
            "I1": nn.I1,
            "I2": nn.I2,
            "mu_ex":mu_ex,
            "sigma_ex":sigma_ex
        },
        "simulation_params": sim_params
    }

filename = "{}/all_params.json".format(op_data_folder)
with open(filename, "w") as f:
        json.dump(data, f, indent=4)
# breakpoint()
PlotAll(input_data_folder=op_data_folder, op_plot_folder=op_plot_folder)
    # print(f"All parameters saved to {filename}")
# last_activity_all=  (last_activity_all > threshold).astype(float)*last_activity_all
# last_activity_ctx_all =  (last_activity_ctx_all > threshold).astype(float)*last_activity_ctx_all
# plot_corr_matrix(last_activity_all[0], fname="{}/corr_matrix.svg".format(op_plot_folder))
# plot_corr_matrix(last_activity_ctx_all[0], fname="{}/corr_matrix_ctx.svg".format(op_plot_folder))
# # plt.plot()

# cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
# xlabs =  [f"{i}" for i in range(N_off_days)]
# Title = "Ensemble similarity"
# # plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr.svg", use_bar_plot=True)
# mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
#     last_activity_all ,                # shape: (sims, time, neurons)
#     ref_time_idx=0,         # Encoding
#     xlabels=xlabs,         # must match number of non-ref times
#     include_ref_bar=True,
#     title="Cell population \n activity correlation",
#     fname="{}/encoding_vs_others_mean_std.svg".format(op_plot_folder),
#     cmap = "Oranges",
#     marker = "^"
# )
# # breakpoint()
# mean_corr_cxt, std_corr_cxt, per_sim_corr_cxt, idx_cxt = plot_mean_std_corr_over_time(
#     last_activity_ctx_all ,                # shape: (sims, time, neurons)
#     ref_time_idx=0,         # Encoding
#     xlabels=xlabs,         # must match number of non-ref times
#     include_ref_bar=True,
#     title="Cell population \n activity correlation",
#     fname="{}/encoding_vs_others_mean_std_ctx.svg".format(op_plot_folder),
#     cmap = "Greens"
# )


# FR_history_th = (FR_history_all > threshold).astype(float)*FR_history_all
# FR_history_th_ctx = (FR_ctx_history_all > threshold).astype(float)*FR_ctx_history_all

# breakpoint()
# timepoints = np.arange(0,total_time,1)*1
# plot_firing_rate(timepoints, FR_op_history_all[:, :, 0],lab = "Output neuron",
#                  xlabel="Time (s)", ylabel="Firing Rate (Hz)", c="r",fname= "{}/OP_neuron_activity.svg".format(op_plot_folder))


# plot_activity_n_excitability_time([FR_history_all[0].T,FR_ctx_history_all[0].T],
#                        titles=['Neuronal Activity (HPC)',
#                                 'Neuronal Activity (CTX)'],
#                        fname="{}/Activity.svg".format(op_plot_folder),
#                        cmaps=['Oranges', 'Greens'])


# plot_activity_n_excitability_time([EX_history_all[0].T,EX_history_ctx_all[0].T],
#                        titles=['Neuronal Excitability (HPC)',
#                                 'Neuronal Excitability (CTX)'],
#                        fname="{}/Excitability_ctx.svg".format(op_plot_folder),
#                        cmaps=['Blues', 'Greens'])

# labs = [f"Day {i+1}" for i in off_days]
# plot_weights_over_time(rec_weights_all[-1,off_days],
#                        titles=  labs,
#                        fname="{}/Rec_w.svg".format(op_plot_folder),
#                        cmaps='gray_r')

# plot_weights_over_time(rec_ctx_weights_all[-1,off_days],
#                        titles=  labs,
#                        fname="{}/Rec_w_ctx.svg".format(op_plot_folder),
#                        cmaps='gray_r')

# plot_weights_over_time(mtl_op_weights_all[-1,off_days],
#                        titles=  labs,
#                        fname="{}/mtl_op_w.svg".format(op_plot_folder),
#                        cmaps='gray_r')
# plot_weights_over_time(ctx_op_weights_all[-1,off_days],
#                        titles=  labs,
#                        fname="{}/ctx_op_w.svg".format(op_plot_folder),
#                        cmaps='gray_r')
# plot_weights_over_time(mtl_ctx_weights_all[-1,off_days],
#                        titles=  labs,
#                        fname="{}/mtl_ctx_w.svg".format(op_plot_folder),
#                        cmaps='gray_r')
# plot_weights_over_time(ctx_mtl_weights_all[-1,off_days],
#                        titles=  labs,
#                        fname="{}/ctx_mtl_w.svg".format(op_plot_folder),
#                        cmaps='gray_r')

# breakpoint()