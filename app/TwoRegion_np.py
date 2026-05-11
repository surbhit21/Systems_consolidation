import matplotlib.pyplot as plt
import json
from LoadNPLot2 import PlotAll
import numpy as np
from plotting_widget import *
from Utilities import average_firing_rates_with_active, ensamble_overlap
from tqdm import trange

start_seed = 5
np.random.seed(start_seed)


def softplus(x):
    """Numerically stable NumPy implementation of the softplus activation."""
    return np.log1p(np.exp(np.clip(x, -500, 500)))


def relu(x):
    """NumPy implementation of the rectified linear activation."""
    return np.maximum(0, x)


class twolayer_FF:
    def __init__(self, n_inp, n_MTL, n_CTX, baseline_e, base_e_ctx, tau=20.0, dt=1.0, act='relu',
                 lr=1/800, decay_r=1/1000,
                 lr_ctx=1/1200, decay_r_ctx=1e-5,
                 lr_mtl_ctx=1/1000, decay_mtl_ctx=1e-5,
                 lr_ctx_mtl=1/1000, decay_ctx_mtl=1e-5,
                 threshold=5,
                 lr_op=1/1000,
                 I0=1, I1=0.05, I2=0.001, Iw=0.01,
                 I0_ctx=1, I1_ctx=0.05, I2_ctx=0.001, Iw_ctx=0.01, tag_threshold=2.):
        # Store population sizes and integration constants.
        self.n_MTL = n_MTL
        self.n_CTX = n_CTX
        self.tau = tau
        self.dt = dt
        
        # Set activation function
        if act == 'relu' or act == relu:
            self.act = relu
        elif act == 'softplus' or act == softplus:
            self.act = softplus
        else:
            self.act = relu  # default
        
        # Learning rates and passive decay constants for each plastic pathway.
        self.lr = lr
        self.decay_r = decay_r
        self.lr_ctx = lr_ctx
        self.decay_r_ctx = decay_r_ctx
        self.lr_mtl_ctx = lr_mtl_ctx
        self.lr_ctx_mtl = lr_ctx_mtl
        self.decay_mtl_ctx = decay_mtl_ctx
        self.decay_ctx_mtl = decay_ctx_mtl
        self.lr_op = lr_op
        
        # Blanket inhibition parameters. I0 is baseline inhibition, while I1/I2
        # scale with total activity and squared activity in each population.
        self.I0 = I0
        self.I1 = I1
        self.I2 = I2
        self.Iw = Iw
        self.I0_ctx = I0_ctx
        self.I1_ctx = I1_ctx
        self.I2_ctx = I2_ctx
        self.Iw_ctx = Iw_ctx
        
        # Baseline excitability values are copied so simulations can reset or
        # modulate them without changing the arrays passed into the constructor.
        self.excitability = baseline_e.copy()
        self.excitability_ctx = base_e_ctx.copy()
        
        # Thresholds
        self.plas_threshold = 0.2
        self.act_threshold = 0
        self.act_threshold_ctx = 0
        self.threshold = threshold
        self.tag_threshold = tag_threshold
        
        # Long-range MTL -> CTX weights start positive and random. Feedback
        # CTX -> MTL weights are initialized to zero by the leading multiplier.
        self.FF_MTL_CTX = (1./np.sqrt(n_CTX)) * np.abs(np.random.normal(0, 1., size=(n_CTX, n_MTL)))
        self.FB_CTX_MTL = 0 * (1./np.sqrt(n_MTL)) * np.abs(np.random.normal(0, 0.05, size=(n_MTL, n_CTX)))
        
        # Plasticity and gain switches are changed by the simulation loop to
        # enable/disable pathways during different phases.
        self.FF_plas = 1.0
        self.gain_FF = 1.0
        self.gain_ctx = 1.
        self.gain_hpc = 1.
        self.op_neuron = 1
        self.on = 1.0
        self.on_ctx = 1.0
        
        # Tags record which neurons became active during a day. The values are
        # used later to selectively boost previously recruited ensembles.
        self.tagged_ACC = np.zeros(n_CTX)
        self.tagged_HPC = np.zeros(n_MTL)
        
        # External input enters MTL through input_w; recurrent weights within
        # MTL and CTX are learned during the simulation.
        self.input_w = np.abs(np.random.normal(0, 0.05, size=(n_inp, n_MTL)))
        self.rec_w = np.zeros((n_MTL, n_MTL))
        self.rec_w_ctx = np.zeros((self.n_CTX, self.n_CTX))
        
        # Output weights
        self.mtl_op_w = 0 * np.abs(np.random.normal(0, 0.05, size=(self.op_neuron, n_MTL)))
        self.ctx_op_w = 0 * np.abs(np.random.normal(0, 0.05, size=(self.op_neuron, n_CTX)))
        
        # Initialize rates
        self.rates_ctx = np.zeros(self.n_CTX)
        self.rates = np.zeros(n_MTL)
        self.op_rate = np.zeros(self.op_neuron)

    def step(self, input_FR, input_FR_ctx, op_FR, op_signal=0, day_c=0, phase=None, ct=1):
        """
        Advance the network by one Euler timestep.

        The method updates MTL, CTX, and output firing rates, applies activity
        tagging, and performs Hebbian updates for the plastic connections.
        """
        # Compute synaptic input to each population before inhibition.
        input_vector = input_FR + self.rec_w .dot(self.rates) - self.FB_CTX_MTL.dot(self.rates_ctx)
        input_CTX = input_FR_ctx + self.rec_w_ctx.dot(self.rates_ctx) + (self.gain_FF * self.FF_MTL_CTX.dot(self.rates))
        input_op = op_FR + self.ctx_op_w.dot(self.rates_ctx) #+ self.mtl_op_w @ self.rates
        
        # Population-wide inhibition grows with total and squared activity.
        I_inhib = self.I0 + self.I1 * np.sum(self.rates) + self.I2 * np.sum(self.rates**2)
        I_inhib_ctx = self.I0_ctx + self.I1_ctx * np.sum(self.rates_ctx) + self.I2_ctx * np.sum(self.rates_ctx**2)
        
        # Tag neurons the first time they exceed the activity threshold. During
        # burnoff, day_c is negative, so tags can be driven below zero if active.
        self.tagged_HPC = self.tagged_HPC + (((self.rates > self.tag_threshold).astype(float) * 
                                               (self.tagged_HPC == 0).astype(float)) * day_c)
        self.tagged_ACC = self.tagged_ACC + (((self.rates_ctx > self.tag_threshold).astype(float) * 
                                               (self.tagged_ACC == 0).astype(float)) * day_c)
        
        # Total input
        input_current = input_vector - I_inhib
        input_ctx = input_CTX - I_inhib_ctx
        
        # Optional debug prints for inspecting activity/inhibition during
        # different simulation phases.
        # if phase == "Encoding" and ct % 500 == 0:
        #     print(input_current.max(), input_ctx.max(), self.rates.max(), self.rates_ctx.max(), 
        #           I_inhib, I_inhib_ctx, input_vector.max(), input_CTX.max(), input_op, phase, ct)
        # if phase == "NREM" and ct % 200 == 0:
        #     print(input_current.max(), input_ctx.max(), self.rates.max(), self.rates_ctx.max(), 
        #           I_inhib, I_inhib_ctx, input_vector.max(), input_CTX.max(), input_op, phase, ct)
        # if phase == "REM" and ct % 500 == 0:
        #     print(input_current.max(), input_ctx.max(), self.rates.max(), self.rates_ctx.max(), 
        #           I_inhib, I_inhib_ctx, input_vector.max(), input_CTX.max(), input_op, phase, ct)

        
        # Rate dynamics
        dr_dt = (-self.rates + self.act(self.gain_hpc * (self.excitability + input_current + self.act_threshold))) / self.tau
        dr_ctx_dt = (-self.rates_ctx + self.act(self.gain_ctx * (self.excitability_ctx + input_ctx + self.act_threshold_ctx))) / self.tau
        dr_op_dt = (-self.op_rate + self.act(input_op)) / self.tau
        
        # Update rates
        self.rates += (dr_dt * self.dt)
        self.rates_ctx += (dr_ctx_dt * self.dt)
        self.op_rate += (dr_op_dt * self.dt)


        # Keep firing rates in a bounded physiological range for numerical stability.
        self.rates = np.clip(self.rates, 0.0, 10.)
        self.rates_ctx = np.clip(self.rates_ctx, 0.0, 10.)
        
        # Hebbian plasticity in MTL recurrent weights, restricted to active units.
        post_mask = self.rates > self.plas_threshold
        heb_dw = self.on * self.lr * np.outer(self.rates * post_mask, self.rates * post_mask) * self.dt
        decay = self.decay_r * self.rec_w * self.dt
        self.rec_w += (heb_dw - decay)
        
        # Plasticity in CTX -> MTL weights
        hebb_dw_ctx_mtl = self.on * self.lr_ctx_mtl * np.outer(self.rates * post_mask, self.rates_ctx* post_mask) * self.dt
        decay_ctx_mtl = self.decay_ctx_mtl * self.FB_CTX_MTL * self.dt
        self.FB_CTX_MTL += (hebb_dw_ctx_mtl - decay_ctx_mtl)
        
        # Plasticity in MTL -> output weights
        # hebb_mtl_op = self.lr_op * op_signal * np.outer(self.op_rate, self.rates * post_mask) * self.dt
        # decay_mtl_op = 1e-8 * self.mtl_op_w * self.dt
        # self.mtl_op_w += (hebb_mtl_op - decay_mtl_op)
        
        # Recompute the active mask for CTX before updating cortical weights.
        post_mask = self.rates_ctx > self.plas_threshold
        hebb_dw_ctx = self.on_ctx * self.lr_ctx * np.outer(self.rates_ctx * post_mask, self.rates_ctx * post_mask) * self.dt
        decay_ctx = self.decay_r_ctx * self.rec_w_ctx * self.dt
        self.rec_w_ctx += (hebb_dw_ctx - decay_ctx)
        
        # Plasticity in MTL -> CTX weights
        hebb_dw = self.FF_plas * self.on_ctx * self.lr_mtl_ctx * np.outer(self.rates_ctx * post_mask, self.rates * post_mask) * self.dt
        decay = self.FF_plas * self.decay_mtl_ctx * self.FF_MTL_CTX * self.dt
        self.FF_MTL_CTX += (hebb_dw - decay)
        
        # Plasticity in CTX -> output weights
        hebb_ctx_op = self.lr_op * op_signal * np.outer(self.op_rate, self.rates_ctx * post_mask) * self.dt
        decay_ctx_op = 1e-8 * self.ctx_op_w * self.dt
        self.ctx_op_w += (hebb_ctx_op - decay_ctx_op)
        
        # Cap the total CTX -> output strength to prevent runaway output drive.
        sum_w = np.sum(self.ctx_op_w)
        if sum_w >= 2:
            self.ctx_op_w *= (2./sum_w)
        self.ctx_op_w = np.clip(self.ctx_op_w, 0., None)
        
        # sum_w = np.sum(self.mtl_op_w)
        # if sum_w > 0:
        #     self.mtl_op_w /= sum_w
        # self.mtl_op_w = np.clip(self.mtl_op_w, 0, None)
        
        # Enforce non-negative bounded synapses after all plasticity updates.
        self.rec_w = np.clip(self.rec_w, 0.0, 1.0)
        self.rec_w_ctx = np.clip(self.rec_w_ctx, 0.0, 1.0)
        self.FF_MTL_CTX = np.clip(self.FF_MTL_CTX, 0.0, 1.)
        self.FB_CTX_MTL = np.clip(self.FB_CTX_MTL, 0.0, 1.)
        
        return self.rates.copy(), self.rates_ctx.copy(), self.op_rate.copy()
    
    def _normalize_input_outgoing(self, target_sum=None, eps=1e-12):
        """Normalize columns so that for each input feature, the outgoing weights sum to target_sum."""
        if target_sum is None:
            target_sum = self.target_out_sum
        col_sums = self.rec_w.sum(axis=1, keepdims=True)
        scale = np.where(col_sums > eps, target_sum / col_sums, np.ones_like(col_sums))
        self.rec_w = self.rec_w * scale


# ---------------------------------------------------------------------------
# Main simulation setup
# ---------------------------------------------------------------------------
# These lists are reinitialized inside each simulation but are declared here to
# make the saved quantities explicit at the top level.
FR_history = []
EX_history = []
rec_weights = []
ff_weights = []
last_activity = []
input_history = []

N_off_days = 11
N_neurons_per_day = 20
n = 10 + (N_off_days) * N_neurons_per_day
n_inp = n
n_ctx = n
E_fl = 2.5
E_fl_ctx = 2.5
E_mod = 2.5
max_e = 10
E_ref = 0.7
threshold = 5.0
off_set = 0

IP_plasticity_limit = 7
condition = ""
 
FC_inp = 15
FC_d0_inp = 15
FC_inp_ctx = 15
input = FC_inp * np.ones(n_inp)
zero_input = np.zeros(n_inp)
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
mtl_op_weights_all = []

dob_HPC = []
dob_ACC = []
off_days = [0, 1, 2, 3, 7, 10]

# Build a descriptive simulation name from blockade and plasticity settings.
if not dob_ACC == []:
    condition += "ACCBlock{}".format(dob_ACC)
if not dob_HPC == []:
    condition += "HPCBlock{}".format(dob_HPC)
else:
    condition +="CNT"
if E_mod == 0.0:
    sim_name = "{}_fast_drift_wo_IP_lowI".format(condition)
else:
    sim_name = "{}_fast_drift_with_limited{}_IP_lowI".format(condition, IP_plasticity_limit)

notes = "2 region model with slow drift due to low excitability boosts in both regions with intrinsic plasticity. ACC neurons that are part of the FC engram get an extra boost in excitability during off days."

# Run independent simulations with shifted random seeds.
for i in trange(NUM_SIM):
    t_series = []
    total_time = 0
    np.random.seed(start_seed + i)
    
    # Draw baseline excitability for MTL/HPC and CTX/ACC populations.
    base_E = np.abs(np.random.normal(mu_ex, sigma_ex, size=(n,)))
    base_e_ctx = np.abs(np.random.normal(mu_ex, sigma_ex, size=(n_ctx,)))
    
    nn = twolayer_FF(n_inp=n_inp, n_MTL=n, n_CTX=n_ctx, baseline_e=base_E.copy(),
                     base_e_ctx=base_e_ctx.copy(), tau=20.0, dt=dt, act='relu',
                     lr=1./3000, decay_r=1./3500,
                     lr_ctx=2e-6, decay_r_ctx=0.,
                     lr_mtl_ctx=1/3000., decay_mtl_ctx=1e-6,
                     lr_ctx_mtl=0., decay_ctx_mtl=0.,
                     threshold=threshold,
                     lr_op=1e-3,
                     I0=5, I1=0.7, I2=0.04,
                     I0_ctx=2., I1_ctx=0.6, I2_ctx=0.04)
    
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
    
    nn.excitability = base_E.copy()
    day = 0
    
    # Prime the first day's neuron cohort with elevated excitability.
    nn.excitability[off_set+(day)*N_neurons_per_day:off_set+(day)*N_neurons_per_day+N_neurons_per_day] += E_fl
    nn.excitability_ctx = base_e_ctx.copy()
    nn.excitability_ctx[(off_set+(day)*N_neurons_per_day):(off_set+(day)*N_neurons_per_day+N_neurons_per_day)] += E_fl_ctx
    
    # Initial silent/burnoff period to settle rates before day 0 stimulation.
    for t in range(ID):
        next_FR, FR_ctx, FR_op = nn.step(zero_input, zero_input, 0, 0, day_c=-1, phase="Burnoff", ct=total_time+t)
        FR_history.append(next_FR)
        FR_history_ctx.append(FR_ctx)
        FR_op_history.append(FR_op)
        EX_history.append(nn.excitability.copy())
        EX_history_ctx.append(nn.excitability_ctx.copy())
        input_history.append(zero_input.copy())
    
    total_time += ID
    t_series.append(total_time)
    
    for day in range(N_off_days):
        day_activity = []
        day_activity_ctx = []

        # Reset daily excitability to baseline, then boost the day's new cohort.
        nn.excitability = base_E.copy()
        nn.excitability[off_set+(day)*N_neurons_per_day:off_set+(day)*N_neurons_per_day+N_neurons_per_day] += E_fl
        nn.excitability_ctx = base_e_ctx.copy()
        new_neurons = range(off_set + day*N_neurons_per_day, off_set + day*N_neurons_per_day + N_neurons_per_day, 1)
        
        # During the intrinsic-plasticity window, previously tagged FC neurons
        # receive an additional CTX excitability boost.
        if day <= IP_plasticity_limit:
            FC_ctx_active_neurons = np.where(nn.tagged_ACC == 1.)[0]
            FC_HPC_active_neurons = np.where(nn.tagged_HPC == 1.)[0]
            nn.excitability_ctx[FC_ctx_active_neurons] += E_mod
            for n1 in new_neurons:
                if n1 not in FC_ctx_active_neurons:
                    nn.excitability_ctx[n1] += E_fl_ctx
        else:
            nn.excitability_ctx[new_neurons] += E_fl_ctx
        
        ctx_inp = FC_inp_ctx

        # Day 0 is encoding with output learning; later days replay the input
        # without output supervision.
        if day == 0:
            op_inp = 5.
            op_learning = 1.
            phase = "Encoding"
            input = FC_d0_inp * np.ones(n_inp)
            ctx_inp = input
        else:
            op_inp = 0.
            op_learning = 0.
            input = FC_inp * np.ones(n_inp)
            ctx_inp = input
            phase = "NREM"
        
        for rep in range(Nrep):
            # Optional day-specific blockade of cortical or hippocampal plasticity.
            if day in dob_ACC:
                nn.on_ctx = 0.0
            else:
                nn.on_ctx = 1.0
            
            if day in dob_HPC:
                nn.on = 0.0
            else:
                nn.on = 1.0

            nn.FF_plas = 1.0
            nn.gain_FF = 1.0
            
            # Stimulation/replay epoch.
            for t in range(t_off):
                next_FR, FR_ctx, FR_op = nn.step(input, ctx_inp, op_inp, op_learning, day_c=day+1, phase=phase, ct=total_time+t)
                FR_history.append(next_FR)
                FR_history_ctx.append(FR_ctx)
                FR_op_history.append(FR_op)
                EX_history.append(nn.excitability.copy())
                EX_history_ctx.append(nn.excitability_ctx.copy())
                input_history.append(input.copy())
            
            day_activity.append(np.mean(FR_history[-t_off//2:], axis=0))
            day_activity_ctx.append(np.mean(FR_history_ctx[-t_off//2:], axis=0))
            total_time += t_off
            t_series.append(total_time)
            
            # Inter-repetition rest interval with no external drive.
            for t in range(IR):
                next_FR, FR_ctx, FR_op = nn.step(zero_input, zero_input, 0, 0, day_c=day+1, phase="IR", ct=total_time)
                FR_history.append(next_FR)
                FR_history_ctx.append(FR_ctx)
                FR_op_history.append(FR_op)
                EX_history.append(nn.excitability.copy())
                EX_history_ctx.append(nn.excitability_ctx.copy())
                input_history.append(zero_input.copy())
            
            total_time += IR
            if rep != Nrep - 1:
                t_series.append(total_time)
        
        nn.FF_plas = 0.0
        nn.gain_FF = 0.0

        # Snapshot weights and final repetition activity at the end of each day.
        rec_weights.append(nn.rec_w.copy())
        rec_ctx_weights.append(nn.rec_w_ctx.copy())
        mtl_ctx_weights.append(nn.FF_MTL_CTX.copy())
        ctx_mtl_weights.append(nn.FB_CTX_MTL.copy())
        mtl_op_weights.append(nn.mtl_op_w.copy())
        ctx_op_weights.append(nn.ctx_op_w.copy())
        
        rep_Activity.append(day_activity)
        last_activity.append(day_activity[-1])
        last_activity_ctx.append(day_activity_ctx[-1])
        
        # Post-day offline interval with feedforward MTL -> CTX drive disabled.
        for t in range(ID):
            next_FR, FR_ctx, FR_op = nn.step(zero_input, zero_input, 0, day_c=day+1, phase="REM", ct=total_time+t)
            FR_history.append(next_FR)
            FR_history_ctx.append(FR_ctx)
            FR_op_history.append(FR_op)
            EX_history.append(nn.excitability.copy())
            EX_history_ctx.append(nn.excitability_ctx.copy())
            input_history.append(zero_input.copy())
        
        total_time += ID
        t_series.append(total_time)
    
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

# Convert per-simulation Python lists to arrays for saving and plotting.
FR_history_all = np.stack(FR_history_all)
FR_ctx_history_all = np.stack(FR_ctx_history_all)
FR_op_history_all = np.stack(FR_op_history_all)
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
op_plot_folder = "./plots/{}".format(sim_name)

import os
os.makedirs(op_data_folder, exist_ok=True)

# Save time series and weight snapshots to disk.
np.save("{}/FR_history.npy".format(op_data_folder), FR_history_all)
np.save("{}/FR_history_ctx.npy".format(op_data_folder), FR_ctx_history_all)
np.save("{}/FR_history_op.npy".format(op_data_folder), FR_op_history_all)
np.save("{}/EX_history.npy".format(op_data_folder), EX_history_all)
np.save("{}/EX_history_ctx.npy".format(op_data_folder), EX_history_ctx_all)
np.save("{}/last_activity.npy".format(op_data_folder), last_activity_all)
np.save("{}/last_activity_ctx.npy".format(op_data_folder), last_activity_ctx_all)
np.save("{}/input_history.npy".format(op_data_folder), input_history_all)
np.save("{}/rec_weights.npy".format(op_data_folder), rec_weights_all)
np.save("{}/mtl_op_weights.npy".format(op_data_folder), mtl_op_weights_all)
np.save("{}/ctx_op_weights.npy".format(op_data_folder), ctx_op_weights_all)
np.save("{}/rec_ctx_weights.npy".format(op_data_folder), rec_ctx_weights_all)
np.save("{}/mtl_ctx_weights.npy".format(op_data_folder), mtl_ctx_weights_all)
np.save("{}/ctx_mtl_weights.npy".format(op_data_folder), ctx_mtl_weights_all)

# Store model and simulation parameters alongside the generated arrays.
sim_params = {
    "n": n,
    "n_inp": n_inp,
    "n_ctx": n_ctx,
    "FC_inp": FC_inp,
    "E_fl": E_fl,
    "E_fl_ctx": E_fl_ctx,
    "E_mod": E_mod,
    "threshold": threshold,
    "ID": ID,
    "N_off_days": N_off_days,
    "t_off": t_off,
    "IR": IR,
    "Nrep": Nrep,
    "start_seed": start_seed,
    "max_e": max_e,
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
        "dob_ACC": dob_ACC,
        "dob_HPC": dob_HPC,
        "n_MTL": nn.n_MTL,
        "tau": nn.tau,
        "dt": nn.dt,
        "lr": nn.lr,
        "decay_r": nn.decay_r,
        "lr_ctx": nn.lr_ctx,
        "decay_r_ctx": nn.decay_r_ctx,
        "lr_op": nn.lr_op,
        "lr_mtl_ctx": nn.lr_mtl_ctx,
        "decay_mtl_ctx": nn.decay_mtl_ctx,
        "lr_ctx_mtl": nn.lr_ctx_mtl,
        "decay_ctx_mtl": nn.decay_ctx_mtl,
        "I0": nn.I0,
        "I1": nn.I1,
        "I2": nn.I2,
        "mu_ex": mu_ex,
        "sigma_ex": sigma_ex
    },
    "simulation_params": sim_params
}

filename = "{}/all_params.json".format(op_data_folder)
with open(filename, "w") as f:
    json.dump(data, f, indent=4)

# Drop into the debugger before plotting, which is useful for inspecting arrays
# interactively after a simulation finishes.
# breakpoint()
PlotAll(input_data_folder=op_data_folder, op_plot_folder=op_plot_folder)
