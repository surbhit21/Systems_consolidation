import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from LoadNPLot2 import PlotAll
import numpy as np
from plotting_widget import *
from Utilities import average_firing_rates_with_active, ensamble_overlap
from tqdm import trange

start_seed = 2026
np.random.seed(start_seed)


def softplus(x):
    """Numerically stable NumPy implementation of the softplus activation."""
    return np.log1p(np.exp(np.clip(x, -500, 500)))


def relu(x):
    """NumPy implementation of the rectified linear activation."""
    return np.maximum(0, x)


def tanh_window(t, duration, ramp_width=6.0, ramp_time=10.0):
    """Smoothly ramp up and down within one stimulation epoch."""
    ramp_up = 0.5 * (1.0 + np.tanh((t - ramp_time) / ramp_width))
    ramp_down = 0.5 * (1.0 + np.tanh((duration - 1 - t - ramp_time) / ramp_width))
    return ramp_up * ramp_down


def stable_random_recurrent(n_neurons, scale=0.05, spectral_radius=0.8):
    """Initialize a non-negative recurrent matrix with controlled spectral radius."""
    weights = np.abs(np.random.normal(0, scale, size=(n_neurons, n_neurons)))
    np.fill_diagonal(weights, 0.0)

    eigvals = np.linalg.eigvals(weights)
    radius = np.max(np.abs(eigvals))
    if radius > 0:
        weights *= spectral_radius / radius
    return weights


class twolayer_FF:
    def __init__(self, n_inp, n_MTL, n_CTX, baseline_e, base_e_ctx, tau=20.0, dt=1.0, act='relu',
                 lr=1/800, decay_r=1/1000,
                 lr_ctx=1/1200, decay_r_ctx=1e-5,
                 lr_ctx_late=None, lr_ctx_late_start_day=2,
                 lr_mtl_ctx=1/1000, decay_mtl_ctx=1e-5,
                 lr_ctx_mtl=1/1000, decay_ctx_mtl=1e-5,
                 threshold=5,
                 lr_op_mtl=1/1000, lr_op_ctx=1/1000,
                 I0=1, I1=0.05, I2=0.001, Iw=0.01,
                 I0_ctx=1, I1_ctx=0.05, I2_ctx=0.001, Iw_ctx=0.01, tag_threshold=5.0,
                 rec_init_scale=0.05, rec_init_radius=0.8,
                 rec_ctx_init_scale=0.05, rec_ctx_init_radius=0.8):
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
        self.lr_ctx_late = lr_ctx if lr_ctx_late is None else lr_ctx_late
        if self.lr_ctx_late < 0:
            raise ValueError("lr_ctx_late must be non-negative")
        if lr_ctx_late_start_day < 0:
            raise ValueError("lr_ctx_late_start_day must be non-negative")
        self.lr_ctx_late_start_day = lr_ctx_late_start_day
        self.lr_mtl_ctx = lr_mtl_ctx
        self.lr_ctx_mtl = lr_ctx_mtl
        self.decay_mtl_ctx = decay_mtl_ctx
        self.decay_ctx_mtl = decay_ctx_mtl
        self.lr_op_mtl = lr_op_mtl
        self.lr_op_ctx = lr_op_ctx
        
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
        self.FF_MTL_CTX = np.clip(
            (1./np.sqrt(n_CTX)) * np.abs(
                np.random.normal(0, 1., size=(n_CTX, n_MTL))
            ),
            0.0,
            1.0,
        )
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
        # Master switch used to freeze every synaptic update (including passive
        # decay) during REM and behavioral recall.
        self.plasticity_on = 1.0
        
        # Tags record which neurons became active during a day. The values are
        # used later to selectively boost previously recruited ensembles.
        self.tagged_ACC = np.zeros(n_CTX)
        self.tagged_HPC = np.zeros(n_MTL)
        
        # External input enters MTL through input_w; recurrent weights within
        # MTL and CTX are learned during the simulation.
        self.input_w = np.abs(np.random.normal(0, 0.05, size=(n_inp, n_MTL)))
        self.rec_w = stable_random_recurrent(
            n_MTL, scale=rec_init_scale, spectral_radius=rec_init_radius
        )
        self.rec_w_ctx = stable_random_recurrent(
            self.n_CTX, scale=rec_ctx_init_scale, spectral_radius=rec_ctx_init_radius
        )
        
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
        input_op = (
            op_FR
            + self.mtl_op_w.dot(self.rates)
            + self.ctx_op_w.dot(self.rates_ctx)
        )
        
        # Population-wide inhibition grows with total and squared activity.
        I_inhib = self.I0 + self.I1 * np.sum(self.rates) + self.I2 * np.sum(self.rates**2)
        I_inhib_ctx = self.I0_ctx + self.I1_ctx * np.sum(self.rates_ctx) + self.I2_ctx * np.sum(self.rates_ctx**2)
        
        # Tag neurons the first time they exceed the activity threshold. During
        # burnoff, day_c is negative, so tags can be driven below zero if active.
        self.tagged_HPC = self.tagged_HPC + (((self.rates >= self.tag_threshold).astype(float) * 
                                               (self.tagged_HPC == 0).astype(float)) * day_c)
        self.tagged_ACC = self.tagged_ACC + (((self.rates_ctx >= self.tag_threshold).astype(float) * 
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
        self.rates = np.clip(self.rates, 0.0, 15.)
        self.rates_ctx = np.clip(self.rates_ctx, 0.0, 15.)
        
        # Hebbian plasticity in MTL recurrent weights, restricted to active units.
        post_mask = self.rates >= self.plas_threshold
        heb_dw = self.on * self.lr * np.outer(self.rates * post_mask, self.rates * post_mask) * self.dt
        decay = self.decay_r * self.rec_w * self.dt
        self.rec_w += self.plasticity_on * (heb_dw - decay)
        
        # Plasticity in CTX -> MTL weights
        hebb_dw_ctx_mtl = self.on * self.lr_ctx_mtl * np.outer(self.rates * post_mask, self.rates_ctx* post_mask) * self.dt
        decay_ctx_mtl = self.decay_ctx_mtl * self.FB_CTX_MTL * self.dt
        self.FB_CTX_MTL += self.plasticity_on * (hebb_dw_ctx_mtl - decay_ctx_mtl)
        
        # Output weights are plastic only when the supervised output-learning
        # signal is present, but both pathways drive the output at all times.
        hebb_mtl_op = self.lr_op_mtl * op_signal * np.outer(self.op_rate, self.rates * post_mask) * self.dt
        decay_mtl_op = op_signal * 1e-8 * self.mtl_op_w * self.dt
        self.mtl_op_w += self.plasticity_on * (hebb_mtl_op - decay_mtl_op)
        
        # Recompute the active mask for CTX before updating cortical weights.
        post_mask = self.rates_ctx >= self.plas_threshold
        # tag_gate = np.ones(self.n_CTX)
        # # tag_gate[self.tagged_ACC == 1.] = 1.0
        # Use the baseline ACC learning rate on days 0 and 1, then switch to the
        # higher rate from lr_ctx_late_start_day onward. day_c is one-based in
        # the main simulation (day_c == day + 1).
        simulation_day = day_c - 1
        effective_lr_ctx = (
            self.lr_ctx_late
            if simulation_day >= self.lr_ctx_late_start_day
            else self.lr_ctx
        )
        ctx_activity = self.rates_ctx * post_mask
        hebb_dw_ctx = (
            self.on_ctx
            * effective_lr_ctx
            * np.outer(ctx_activity, ctx_activity)
            * self.dt
        )
        decay_ctx = self.decay_r_ctx * self.rec_w_ctx * self.dt
        self.rec_w_ctx += self.plasticity_on * (hebb_dw_ctx - decay_ctx)
        
        # Plasticity in MTL -> CTX weights
        hebb_dw = self.FF_plas * self.on_ctx * self.lr_mtl_ctx * np.outer(self.rates_ctx * post_mask, self.rates * post_mask) * self.dt
        decay = self.FF_plas * self.decay_mtl_ctx * self.FF_MTL_CTX * self.dt
        self.FF_MTL_CTX += self.plasticity_on * (hebb_dw - decay)
        
        # Plasticity in CTX -> output weight
        hebb_ctx_op = self.lr_op_ctx * op_signal * np.outer(self.op_rate, self.rates_ctx * post_mask) * self.dt
        decay_ctx_op = op_signal * 1e-8 * self.ctx_op_w * self.dt
        self.ctx_op_w += self.plasticity_on * (hebb_ctx_op - decay_ctx_op)

        # Normalize each output neuron's incoming HPC and CTX weights
        # independently during supervised output learning.
        if op_signal:
            self.mtl_op_w = np.clip(self.mtl_op_w, 0., None)
            self.ctx_op_w = np.clip(self.ctx_op_w, 0., None)

            mtl_op_sums = np.sum(self.mtl_op_w, axis=1, keepdims=True)
            ctx_op_sums = np.sum(self.ctx_op_w, axis=1, keepdims=True)
            self.mtl_op_w = np.divide(
                self.mtl_op_w,
                mtl_op_sums,
                out=np.zeros_like(self.mtl_op_w),
                where=mtl_op_sums > 0,
            )
            self.ctx_op_w = np.divide(
                self.ctx_op_w,
                ctx_op_sums,
                out=np.zeros_like(self.ctx_op_w),
                where=ctx_op_sums > 0,
            )
        
        # Enforce bounds only when plasticity is active. With plasticity off,
        # every synaptic matrix remains bit-for-bit unchanged.
        if self.plasticity_on:
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


def snapshot_weights(model):
    return {
        "rec_w": model.rec_w.copy(),
        "rec_w_ctx": model.rec_w_ctx.copy(),
        "FF_MTL_CTX": model.FF_MTL_CTX.copy(),
        "FB_CTX_MTL": model.FB_CTX_MTL.copy(),
        "mtl_op_w": model.mtl_op_w.copy(),
        "ctx_op_w": model.ctx_op_w.copy(),
    }


def restore_weights(model, weights):
    model.rec_w = weights["rec_w"].copy()
    model.rec_w_ctx = weights["rec_w_ctx"].copy()
    model.FF_MTL_CTX = weights["FF_MTL_CTX"].copy()
    model.FB_CTX_MTL = weights["FB_CTX_MTL"].copy()
    model.mtl_op_w = weights["mtl_op_w"].copy()
    model.ctx_op_w = weights["ctx_op_w"].copy()


def reset_positive_weight_changes(weights, reference_weights):
    """Reset only positive weight changes relative to a reference matrix.

    This implements an LTP-erasure-like manipulation: synapses that potentiated
    above the last non-erased reference are returned to that reference value,
    while unchanged or depressed synapses are left untouched.
    """
    delta_w = weights - reference_weights
    potentiated_mask = delta_w > 0
    erased_weights = weights.copy()
    erased_weights[potentiated_mask] = reference_weights[potentiated_mask]
    erased_weights = np.clip(erased_weights, 0.0, 1.0)
    return erased_weights, potentiated_mask


# ---------------------------------------------------------------------------
# Main simulation setup
# ---------------------------------------------------------------------------
# These lists are reinitialized inside each simulation but are declared here to
# make the saved quantities explicit at the top level.


def main():
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
    E_fl = 1.7
    E_fl_ctx = 1.7
    E_mod = 8.
    max_e = 10
    E_ref = 0.7
    threshold = 5.0
    off_set = 0

    IP_plasticity_limit = N_off_days
    tau_IE = 2.0
    condition = ""
 
    FC_inp = 13
    FC_d0_inp = 13
    FC_inp_ctx = 13
    input = FC_inp * np.ones(n_inp)
    zero_input = np.zeros(n_inp)
    mu_ex = 0
    sigma_ex = 1.0

    rem_duration = 1000
    nrem_duration = 200
    nrem_reactivation_fraction = 0.75
    # ID remains the total post-learning duration for compatibility with the
    # existing plotting utilities.
    ID = rem_duration + nrem_duration
    dt = 1
    NUM_SIM = 10
    t_off = 100
    input_ramp_time = 10.0
    input_ramp_width = 6.0
    IR = 100
    Nrep = 10
    N_recall = 1
    recall_t = t_off
    recall_IR = IR
    recall_days = list(range(N_off_days))
    total_time = 0
    input_history_all = []

    FR_history_all = []
    FR_ctx_history_all = []
    FR_op_history_all = []
    last_activity_all = []
    last_activity_ctx_all = []
    recall_activity_all = []
    recall_activity_ctx_all = []
    recall_op_activity_all = []

    EX_history_all = []
    EX_history_ctx_all = []

    rec_weights_all = []
    rec_ctx_weights_all = []
    hpc_ltp_erasure_counts_all = []
    acc_ltp_erasure_counts_all = []

    mtl_ctx_weights_all = []
    ctx_mtl_weights_all = []

    ctx_op_weights_all = []
    mtl_op_weights_all = []

    dob_HPC = []
    dob_ACC = []
    ip_block_days = []
    hpc_ltp_erasure_days = []
    acc_ltp_erasure_days = [0]
    off_days = [0, 1, 2, 3, 7, 10]

    # Build a descriptive simulation name from blockade and plasticity settings.
    if not dob_ACC == []:
        condition += "ACCBlock{}_learningOffline".format(dob_ACC)
    if not dob_HPC == []:
        condition += "HPCBlock{}_learningOffline".format(dob_HPC)
    if not ip_block_days == []:
        condition += "IPBlock{}_learningOffline".format(ip_block_days)
    if not hpc_ltp_erasure_days == []:
        condition += "HPCLTPErase{}_learningOffline".format(hpc_ltp_erasure_days)
    if not acc_ltp_erasure_days == []:
        condition += "ACCLTPErase{}_learningOffline".format(acc_ltp_erasure_days)
    if condition == "":
        condition += "CNT_learningOffline"
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
                         lr=1./800, decay_r=1./1000,
                         lr_ctx=1e-5, decay_r_ctx=1e-8,
                         lr_ctx_late=1e-4, lr_ctx_late_start_day=2,
                         lr_mtl_ctx=1e-3, decay_mtl_ctx=1e-6,
                         lr_ctx_mtl=0., decay_ctx_mtl=0.,
                         threshold=threshold,
                         lr_op_mtl=1e-3, lr_op_ctx=2e-3,
                         I0=9., I1=0.5, I2=0.05,
                         I0_ctx=9., I1_ctx=0.5, I2_ctx=0.05)
    
        input_history = []
        FR_history = []
        FR_history_ctx = []
        FR_op_history = []
        last_activity = []
        last_activity_ctx = []
        recall_activity = []
        recall_activity_ctx = []
        recall_op_activity = []
        EX_history = []
        EX_history_ctx = []
        rec_weights = []
        rec_ctx_weights = []
        hpc_ltp_erasure_counts = []
        acc_ltp_erasure_counts = []
        mtl_ctx_weights = []
        ctx_mtl_weights = []
        rep_Activity = []
        high_threshold = 5
        mtl_op_weights = []
        ctx_op_weights = []
        hpc_ltp_reference_weights = nn.rec_w.copy()
        previous_day_hpc_ltp_erased = False
        acc_ltp_reference_weights = nn.rec_w_ctx.copy()
        acc_ltp_reference_ff_weights = nn.FF_MTL_CTX.copy()
        previous_day_acc_ltp_erased = False
    
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
            erase_hpc_ltp = day in hpc_ltp_erasure_days
            erase_acc_ltp = day in acc_ltp_erasure_days
            if erase_hpc_ltp and not previous_day_hpc_ltp_erased:
                hpc_ltp_reference_weights = nn.rec_w.copy()
            elif not erase_hpc_ltp and not previous_day_hpc_ltp_erased:
                hpc_ltp_reference_weights = nn.rec_w.copy()
            if erase_acc_ltp and not previous_day_acc_ltp_erased:
                acc_ltp_reference_weights = nn.rec_w_ctx.copy()
                acc_ltp_reference_ff_weights = nn.FF_MTL_CTX.copy()
            elif not erase_acc_ltp and not previous_day_acc_ltp_erased:
                acc_ltp_reference_weights = nn.rec_w_ctx.copy()
                acc_ltp_reference_ff_weights = nn.FF_MTL_CTX.copy()

            # Reset daily excitability to baseline, then boost the day's new cohort.
            nn.excitability = base_E.copy()
            nn.excitability[off_set+(day)*N_neurons_per_day:off_set+(day)*N_neurons_per_day+N_neurons_per_day] += E_fl
            nn.excitability_ctx = base_e_ctx.copy()
            new_neurons = range(off_set + day*N_neurons_per_day, off_set + day*N_neurons_per_day + N_neurons_per_day, 1)
        
            # During the intrinsic-plasticity window, every tagged ACC neuron gets
            # an excitability boost kernel starting from the day it was tagged.
            tagged_ACC_neurons = np.where(nn.tagged_ACC == 1.)[0]
            if len(tagged_ACC_neurons) > 0:
                tagged_days = nn.tagged_ACC[tagged_ACC_neurons] - 1
                days_since_tag = day - tagged_days
                active_kernel = (days_since_tag >= 0) & (days_since_tag <= IP_plasticity_limit)
                boosted_ACC_neurons = tagged_ACC_neurons[active_kernel]
                ie_gain = E_mod * np.exp(-days_since_tag[active_kernel] / tau_IE)
                if day not in ip_block_days:
                    nn.excitability_ctx[boosted_ACC_neurons] += ie_gain
            else:
                boosted_ACC_neurons = np.array([], dtype=int)

            tagged_ACC_set = set(tagged_ACC_neurons)
            if day <= IP_plasticity_limit:
                for n1 in new_neurons:
                    if n1 not in tagged_ACC_set:
                        nn.excitability_ctx[n1] += E_fl_ctx
            else:
                nn.excitability_ctx[new_neurons] += E_fl_ctx
        
            # breakpoint()
            ctx_inp = FC_inp_ctx

            # Day 0 is encoding with output learning; later days replay the input
            # without output supervision.
            if day == 0:
                op_inp = 1.
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

            # For blockade days, let the network run with its usual plasticity
            # dynamics and then restore the LTP-like weights to their pre-block
            # values at the end of the day.
            block_ACC = day in dob_ACC
            block_HPC = day in dob_HPC
            pre_block_rec_ctx = nn.rec_w_ctx.copy() if block_ACC else None
            pre_block_mtl_ctx = nn.FF_MTL_CTX.copy() if block_ACC else None
            pre_block_rec_w = nn.rec_w.copy() if block_HPC else None
            pre_block_ctx_mtl = nn.FB_CTX_MTL.copy() if block_HPC else None
            hpc_ltp_erasure_count = 0
            acc_ltp_erasure_count = 0
        
            for rep in range(Nrep):
                # Keep learning switches on; blockade is applied by restoring the
                # affected LTP-like weights after the day's stimulation/replay.
                nn.on_ctx = 1.0
                nn.on = 1.0

                nn.FF_plas = 1.0
                nn.gain_FF = 1.0
            
                # Stimulation/replay epoch.
                for t in range(t_off):
                    ramp = tanh_window(t, t_off, ramp_width=input_ramp_width, ramp_time=input_ramp_time)
                    ramped_input = ramp * input
                    ramped_ctx_inp = ramp * ctx_inp
                    ramped_op_inp = ramp * op_inp
                    next_FR, FR_ctx, FR_op = nn.step(
                        ramped_input,
                        ramped_ctx_inp,
                        ramped_op_inp,
                        op_learning,
                        day_c=day+1,
                        phase=phase,
                        ct=total_time+t,
                    )
                    FR_history.append(next_FR)
                    FR_history_ctx.append(FR_ctx)
                    FR_op_history.append(FR_op)
                    EX_history.append(nn.excitability.copy())
                    EX_history_ctx.append(nn.excitability_ctx.copy())
                    input_history.append(ramped_input.copy())
            
                day_activity.append(np.mean(FR_history[-t_off:], axis=0))
                day_activity_ctx.append(np.mean(FR_history_ctx[-t_off:], axis=0))
                total_time += t_off
                t_series.append(total_time)
            
                # the learned weight changes are reversed at 
                # the end of the day to simulate the effect of
                #  a pharmacological blockade that prevents 
                # LTP during that day's replay, while allowing normal 
                # plasticity during other phases. 
                # This way we can isolate the contribution of potentiation 
                # in each pathway to consolidation without affecting other aspects 
                # of network dynamics during the day.
            
                # if block_ACC:
                #     nn.rec_w_ctx = pre_block_rec_ctx
                #     nn.FF_MTL_CTX = pre_block_mtl_ctx
                # if block_HPC:
                #     nn.rec_w = pre_block_rec_w
                #     nn.FB_CTX_MTL = pre_block_ctx_mtl
                if erase_hpc_ltp:
                    nn.rec_w, hpc_ltp_erasure_mask = reset_positive_weight_changes(
                        nn.rec_w,
                        hpc_ltp_reference_weights,
                    )
                    hpc_ltp_erasure_count += int(np.sum(hpc_ltp_erasure_mask))
                if erase_acc_ltp:
                    nn.rec_w_ctx, acc_ltp_erasure_mask = reset_positive_weight_changes(
                        nn.rec_w_ctx,
                        acc_ltp_reference_weights,
                    )
                    nn.FF_MTL_CTX, acc_ff_ltp_erasure_mask = reset_positive_weight_changes(
                        nn.FF_MTL_CTX,
                        acc_ltp_reference_ff_weights,
                    )
                    acc_ltp_erasure_count += int(
                        np.sum(acc_ltp_erasure_mask) + np.sum(acc_ff_ltp_erasure_mask)
                    )
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

            if erase_hpc_ltp:
                hpc_ltp_erasure_counts.append(hpc_ltp_erasure_count)
                previous_day_hpc_ltp_erased = True
            else:
                hpc_ltp_erasure_counts.append(0)
                hpc_ltp_reference_weights = nn.rec_w.copy()
                previous_day_hpc_ltp_erased = False

            if erase_acc_ltp:
                acc_ltp_erasure_counts.append(acc_ltp_erasure_count)
                previous_day_acc_ltp_erased = True
            else:
                acc_ltp_erasure_counts.append(0)
                acc_ltp_reference_weights = nn.rec_w_ctx.copy()
                acc_ltp_reference_ff_weights = nn.FF_MTL_CTX.copy()
                previous_day_acc_ltp_erased = False

            # breakpoint()
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

            if day in recall_days:
                # Scheduled recall test: present the conditioned stimulus with all
                # weights frozen, then restore the pre-recall network state so this
                # readout does not perturb later consolidation days.
                pre_recall_weights = snapshot_weights(nn)
                pre_recall_rates = nn.rates.copy()
                pre_recall_rates_ctx = nn.rates_ctx.copy()
                pre_recall_op_rate = nn.op_rate.copy()
                pre_recall_excitability = nn.excitability.copy()
                pre_recall_excitability_ctx = nn.excitability_ctx.copy()
                pre_recall_on = nn.on
                pre_recall_on_ctx = nn.on_ctx
                pre_recall_ff_plas = nn.FF_plas
                pre_recall_gain_ff = nn.gain_FF

                nn.excitability = base_E.copy()
                nn.excitability_ctx = base_e_ctx.copy()
                nn.on = 0.0
                nn.on_ctx = 0.0
                nn.FF_plas = 0.0
                nn.gain_FF = 1.0
                recall_input = FC_inp * np.ones(n_inp)
                recall_ctx_input = recall_input.copy()

                for rep in range(N_recall):
                    recall_FR = []
                    recall_FR_ctx = []
                    recall_FR_op = []

                    for t in range(recall_t):
                        ramp = tanh_window(t, recall_t, ramp_width=input_ramp_width, ramp_time=input_ramp_time)
                        ramped_recall_input = ramp * recall_input
                        ramped_recall_ctx_input = ramp * recall_ctx_input
                        next_FR, FR_ctx, FR_op = nn.step(
                            ramped_recall_input,
                            ramped_recall_ctx_input,
                            0,
                            0,
                            day_c=0,
                            phase="Recall",
                            ct=total_time+t,
                        )
                        restore_weights(nn, pre_recall_weights)
                        recall_FR.append(next_FR)
                        recall_FR_ctx.append(FR_ctx)
                        recall_FR_op.append(FR_op)

                    recall_activity.append(np.mean(recall_FR[-recall_t//2:], axis=0))
                    recall_activity_ctx.append(np.mean(recall_FR_ctx[-recall_t//2:], axis=0))
                    recall_op_activity.append(np.mean(recall_FR_op[-recall_t//2:], axis=0))

                    if rep != N_recall - 1:
                        for t in range(recall_IR):
                            nn.step(
                                zero_input,
                                zero_input,
                                0,
                                0,
                                day_c=0,
                                phase="RecallIR",
                                ct=total_time+t,
                            )
                            restore_weights(nn, pre_recall_weights)

                restore_weights(nn, pre_recall_weights)
                nn.rates = pre_recall_rates
                nn.rates_ctx = pre_recall_rates_ctx
                nn.op_rate = pre_recall_op_rate
                nn.excitability = pre_recall_excitability
                nn.excitability_ctx = pre_recall_excitability_ctx
                nn.on = pre_recall_on
                nn.on_ctx = pre_recall_on_ctx
                nn.FF_plas = pre_recall_ff_plas
                nn.gain_FF = pre_recall_gain_ff
    
        input_history_all.append(input_history)
        FR_history_all.append(FR_history)
        FR_ctx_history_all.append(FR_history_ctx)
        FR_op_history_all.append(FR_op_history)
        last_activity_all.append(last_activity)
        last_activity_ctx_all.append(last_activity_ctx)
        recall_activity_all.append(recall_activity)
        recall_activity_ctx_all.append(recall_activity_ctx)
        recall_op_activity_all.append(recall_op_activity)
        EX_history_all.append(EX_history)
        EX_history_ctx_all.append(EX_history_ctx)
        rec_weights_all.append(rec_weights)
        rec_ctx_weights_all.append(rec_ctx_weights)
        hpc_ltp_erasure_counts_all.append(hpc_ltp_erasure_counts)
        acc_ltp_erasure_counts_all.append(acc_ltp_erasure_counts)
        mtl_ctx_weights_all.append(mtl_ctx_weights)
        ctx_mtl_weights_all.append(ctx_mtl_weights)
        mtl_op_weights_all.append(mtl_op_weights)
        ctx_op_weights_all.append(ctx_op_weights)
        print("ACC tagged:", (nn.tagged_ACC == 1.).sum(), "HPC tagged:", (nn.tagged_HPC == 1.).sum())
    # Convert per-simulation Python lists to arrays for saving and plotting.
    FR_history_all = np.stack(FR_history_all)
    FR_ctx_history_all = np.stack(FR_ctx_history_all)
    FR_op_history_all = np.stack(FR_op_history_all)
    input_history = np.stack(input_history)
    EX_history_all = np.stack(EX_history_all)
    EX_history_ctx_all = np.stack(EX_history_ctx_all)
    last_activity_all = np.stack(last_activity_all)
    last_activity_ctx_all = np.stack(last_activity_ctx_all)
    recall_activity_all = np.stack(recall_activity_all)
    recall_activity_ctx_all = np.stack(recall_activity_ctx_all)
    recall_op_activity_all = np.stack(recall_op_activity_all)
    rec_weights_all = np.stack(rec_weights_all)
    rec_ctx_weights_all = np.stack(rec_ctx_weights_all)
    hpc_ltp_erasure_counts_all = np.stack(hpc_ltp_erasure_counts_all)
    acc_ltp_erasure_counts_all = np.stack(acc_ltp_erasure_counts_all)
    mtl_ctx_weights_all = np.stack(mtl_ctx_weights_all)
    ctx_mtl_weights_all = np.stack(ctx_mtl_weights_all)
    mtl_op_weights_all = np.stack(mtl_op_weights_all)
    ctx_op_weights_all = np.stack(ctx_op_weights_all)

    op_data_folder = "./data/{}".format(sim_name)
    op_plot_folder = "./plots/{}".format(sim_name)
    # breakpoint()
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
    np.save("{}/recall_activity.npy".format(op_data_folder), recall_activity_all)
    np.save("{}/recall_activity_ctx.npy".format(op_data_folder), recall_activity_ctx_all)
    np.save("{}/recall_activity_op.npy".format(op_data_folder), recall_op_activity_all)
    np.save("{}/input_history.npy".format(op_data_folder), input_history_all)
    np.save("{}/rec_weights.npy".format(op_data_folder), rec_weights_all)
    np.save("{}/mtl_op_weights.npy".format(op_data_folder), mtl_op_weights_all)
    np.save("{}/ctx_op_weights.npy".format(op_data_folder), ctx_op_weights_all)
    np.save("{}/rec_ctx_weights.npy".format(op_data_folder), rec_ctx_weights_all)
    np.save("{}/mtl_ctx_weights.npy".format(op_data_folder), mtl_ctx_weights_all)
    np.save("{}/ctx_mtl_weights.npy".format(op_data_folder), ctx_mtl_weights_all)
    np.save("{}/hpc_ltp_erasure_counts.npy".format(op_data_folder), hpc_ltp_erasure_counts_all)
    np.save("{}/acc_ltp_erasure_counts.npy".format(op_data_folder), acc_ltp_erasure_counts_all)
    
    # Store model and simulation parameters alongside the generated arrays.
    sim_params = {
        "n": n,
        "n_inp": n_inp,
        "n_ctx": n_ctx,
        "FC_inp": FC_inp,
        "E_fl": E_fl,
        "E_fl_ctx": E_fl_ctx,
        "E_mod": E_mod,
        "tau_IE": tau_IE,
        "threshold": threshold,
        "ID": ID,
        "N_off_days": N_off_days,
        "t_off": t_off,
        "input_ramp_time": input_ramp_time,
        "input_ramp_width": input_ramp_width,
        "IR": IR,
        "Nrep": Nrep,
        "N_recall": N_recall,
        "recall_t": recall_t,
        "recall_IR": recall_IR,
        "recall_days": list(recall_days),
        "start_seed": start_seed,
        "max_e": max_e,
        "total_time": total_time,
        "dt": dt,
        "NUM_SIM": NUM_SIM,
        "notes": notes,
        "off_days": off_days,
        "t_series": t_series,
        "IP_plasticity_limit": IP_plasticity_limit,
        "ip_block_days": list(ip_block_days),
        "hpc_ltp_erasure_days": list(hpc_ltp_erasure_days),
        "acc_ltp_erasure_days": list(acc_ltp_erasure_days),
        "freezing_fr_max":10.0
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
            "lr_ctx_late": nn.lr_ctx_late,
            "lr_ctx_late_start_day": nn.lr_ctx_late_start_day,
            "lr_op_mtl": nn.lr_op_mtl,
            "lr_op_ctx": nn.lr_op_ctx,
            "lr_mtl_ctx": nn.lr_mtl_ctx,
            "decay_mtl_ctx": nn.decay_mtl_ctx,
            "lr_ctx_mtl": nn.lr_ctx_mtl,
            "decay_ctx_mtl": nn.decay_ctx_mtl,
            "I0": nn.I0,
            "I1": nn.I1,
            "I2": nn.I2,
            "mu_ex": mu_ex,
            "sigma_ex": sigma_ex,
            "tag_threshold": nn.tag_threshold
        },
        "simulation_params": sim_params
    }

    filename = "{}/all_params.json".format(op_data_folder)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    PlotAll(input_data_folder=op_data_folder, op_plot_folder=op_plot_folder)


def main_learning_offline():
    """Run explicit learning, REM, NREM-reactivation, and recall phases."""
    import os

    n_days = 11
    neurons_per_day = 20
    n = 10 + n_days * neurons_per_day
    n_inp = n
    n_ctx = n
    dt = 1
    num_sim = 10

    learning_presentations = 10
    learning_duration = 100
    learning_ir = 100
    rem_duration = 1000
    nrem_duration = 200
    nrem_reactivation_fraction = 0.75
    recall_duration = 100

    fc_input = 13.0
    day0_input = 13.0
    excitability_boost = 1.7
    acc_excitability_boost = 1.7
    tagged_acc_boost = 8.0
    tagged_acc_tau = 2.0
    ip_limit = n_days
    ramp_time = 10.0
    ramp_width = 6.0
    freezing_fr_max = 10.0

    sim_name = "CNT_learning_REM_NREM75_recall"
    data_folder = "./data/{}".format(sim_name)
    plot_folder = "./plots/{}".format(sim_name)
    os.makedirs(data_folder, exist_ok=True)

    all_hpc = []
    all_acc = []
    all_output = []
    all_input = []
    all_ex_hpc = []
    all_ex_acc = []
    all_last_hpc = []
    all_last_acc = []
    all_recall_hpc = []
    all_recall_acc = []
    all_recall_output = []
    all_recall_freezing = []
    all_nrem_masks = []
    all_rec_hpc = []
    all_rec_acc = []
    all_hpc_acc = []
    all_acc_hpc = []
    all_hpc_output_w = []
    all_acc_output_w = []
    final_model = None
    final_total_time = 0

    zero_input = np.zeros(n_inp)
    n_reactivated = int(round(n_inp * nrem_reactivation_fraction))

    for sim in trange(num_sim):
        np.random.seed(start_seed + sim)
        base_hpc = np.abs(np.random.normal(0.0, 1.0, size=n))
        base_acc = np.abs(np.random.normal(0.0, 1.0, size=n_ctx))
        model = twolayer_FF(
            n_inp=n_inp,
            n_MTL=n,
            n_CTX=n_ctx,
            baseline_e=base_hpc.copy(),
            base_e_ctx=base_acc.copy(),
            tau=20.0,
            dt=dt,
            act="relu",
            lr=1.0 / 800,
            decay_r=1.0 / 1000,
            lr_ctx=1e-5,
            decay_r_ctx=1e-8,
            lr_ctx_late=1e-4,
            lr_ctx_late_start_day=2,
            lr_mtl_ctx=1e-3,
            decay_mtl_ctx=1e-6,
            lr_ctx_mtl=0.0,
            decay_ctx_mtl=0.0,
            threshold=5.0,
            lr_op_mtl=1e-3,
            lr_op_ctx=2e-3,
            I0=9.0,
            I1=0.5,
            I2=0.05,
            I0_ctx=9.0,
            I1_ctx=0.5,
            I2_ctx=0.05,
        )

        hpc_history = []
        acc_history = []
        output_history = []
        input_history = []
        ex_hpc_history = []
        ex_acc_history = []
        last_hpc = []
        last_acc = []
        recall_hpc = []
        recall_acc = []
        recall_output = []
        recall_freezing = []
        nrem_masks = []
        rec_hpc = []
        rec_acc = []
        hpc_acc = []
        acc_hpc = []
        hpc_output_w = []
        acc_output_w = []
        total_time = 0

        def record_step(hpc_rates, acc_rates, output_rates, external_input):
            hpc_history.append(hpc_rates)
            acc_history.append(acc_rates)
            output_history.append(output_rates)
            input_history.append(external_input.copy())
            ex_hpc_history.append(model.excitability.copy())
            ex_acc_history.append(model.excitability_ctx.copy())

        # Initial settling is non-plastic and is not part of a sleep phase.
        model.plasticity_on = 0.0
        for t in range(1000):
            rates = model.step(
                zero_input, zero_input, 0, 0,
                day_c=-1, phase="Burnoff", ct=total_time + t,
            )
            record_step(*rates, zero_input)
        total_time += 1000

        for day in range(n_days):
            model.excitability = base_hpc.copy()
            model.excitability_ctx = base_acc.copy()
            cohort_start = day * neurons_per_day
            cohort_stop = cohort_start + neurons_per_day
            model.excitability[cohort_start:cohort_stop] += excitability_boost
            model.excitability_ctx[cohort_start:cohort_stop] += acc_excitability_boost

            tagged = np.where(model.tagged_ACC > 0)[0]
            if tagged.size:
                tagged_days = model.tagged_ACC[tagged] - 1
                age = day - tagged_days
                active = (age >= 0) & (age <= ip_limit)
                model.excitability_ctx[tagged[active]] += (
                    tagged_acc_boost * np.exp(-age[active] / tagged_acc_tau)
                )

            # Learning occurs only once, on day 0. Later days skip this block
            # entirely and contain only offline REM/NREM followed by recall.
            model.plasticity_on = 1.0
            model.on = 1.0
            model.on_ctx = 1.0
            model.FF_plas = 1.0
            model.gain_FF = 1.0
            learning_input = (
                day0_input if day == 0 else fc_input
            ) * np.ones(n_inp)
            output_input = 1.0 if day == 0 else 0.0
            output_learning = 1.0 if day == 0 else 0.0
            learning_hpc = []
            learning_acc = []
            presentations_today = learning_presentations if day == 0 else 0

            for _ in range(presentations_today):
                for t in range(learning_duration):
                    ramp = tanh_window(
                        t, learning_duration,
                        ramp_width=ramp_width,
                        ramp_time=ramp_time,
                    )
                    driven_input = ramp * learning_input
                    rates = model.step(
                        driven_input,
                        driven_input,
                        ramp * output_input,
                        output_learning,
                        day_c=day + 1,
                        phase="Learning",
                        ct=total_time + t,
                    )
                    record_step(*rates, driven_input)
                    learning_hpc.append(rates[0])
                    learning_acc.append(rates[1])
                total_time += learning_duration

                for t in range(learning_ir):
                    rates = model.step(
                        zero_input, zero_input, 0, 0,
                        day_c=day + 1,
                        phase="LearningIR",
                        ct=total_time + t,
                    )
                    record_step(*rates, zero_input)
                total_time += learning_ir

            # REM: no external drive and absolutely no synaptic plasticity.
            model.plasticity_on = 0.0
            model.gain_FF = 0.0
            for t in range(rem_duration):
                rates = model.step(
                    zero_input, zero_input, 0, 0,
                    day_c=day + 1, phase="REM", ct=total_time + t,
                )
                record_step(*rates, zero_input)
            total_time += rem_duration

            # NREM: reactivate a random 75% of input/neuron channels and allow
            # plasticity. A fresh subset is sampled independently on each day.
            chosen = np.random.choice(n_inp, size=n_reactivated, replace=False)
            nrem_mask = np.zeros(n_inp, dtype=bool)
            nrem_mask[chosen] = True
            nrem_masks.append(nrem_mask)
            nrem_input = fc_input * nrem_mask.astype(float)
            model.plasticity_on = 1.0
            model.gain_FF = 1.0
            nrem_hpc_steps = []
            nrem_acc_steps = []
            for t in range(nrem_duration):
                ramp = tanh_window(
                    t, nrem_duration,
                    ramp_width=ramp_width,
                    ramp_time=ramp_time,
                )
                driven_input = ramp * nrem_input
                rates = model.step(
                    driven_input, driven_input, 0, 0,
                    day_c=day + 1, phase="NREM", ct=total_time + t,
                )
                record_step(*rates, driven_input)
                nrem_hpc_steps.append(rates[0])
                nrem_acc_steps.append(rates[1])
            total_time += nrem_duration

            # Daily activity summaries now describe offline reactivation on
            # every day, rather than nonexistent learning trials after day 0.
            last_hpc.append(np.mean(nrem_hpc_steps, axis=0))
            last_acc.append(np.mean(nrem_acc_steps, axis=0))

            rec_hpc.append(model.rec_w.copy())
            rec_acc.append(model.rec_w_ctx.copy())
            hpc_acc.append(model.FF_MTL_CTX.copy())
            acc_hpc.append(model.FB_CTX_MTL.copy())
            hpc_output_w.append(model.mtl_op_w.copy())
            acc_output_w.append(model.ctx_op_w.copy())

            # Dedicated standardized recall: full cue, zero output teaching
            # signal, and all synaptic plasticity/decay frozen. Recall starts
            # from silent rates and is restored afterward so it cannot affect
            # the next learning day.
            saved_weights = snapshot_weights(model)
            saved_rates = model.rates.copy()
            saved_acc_rates = model.rates_ctx.copy()
            saved_output_rate = model.op_rate.copy()
            saved_hpc_ex = model.excitability.copy()
            saved_acc_ex = model.excitability_ctx.copy()
            model.plasticity_on = 0.0
            model.gain_FF = 1.0
            model.excitability = base_hpc.copy()
            model.excitability_ctx = base_acc.copy()
            model.rates.fill(0.0)
            model.rates_ctx.fill(0.0)
            model.op_rate.fill(0.0)
            recall_hpc_steps = []
            recall_acc_steps = []
            recall_output_steps = []
            full_recall_input = fc_input * np.ones(n_inp)
            for t in range(recall_duration):
                ramp = tanh_window(
                    t, recall_duration,
                    ramp_width=ramp_width,
                    ramp_time=ramp_time,
                )
                driven_input = ramp * full_recall_input
                rates = model.step(
                    driven_input, driven_input, 0, 0,
                    day_c=day + 1, phase="Recall", ct=total_time + t,
                )
                recall_hpc_steps.append(rates[0])
                recall_acc_steps.append(rates[1])
                recall_output_steps.append(rates[2])

            mean_recall_hpc = np.mean(recall_hpc_steps[-recall_duration // 2:], axis=0)
            mean_recall_acc = np.mean(recall_acc_steps[-recall_duration // 2:], axis=0)
            mean_recall_output = np.mean(
                recall_output_steps[-recall_duration // 2:], axis=0
            )
            recall_hpc.append(mean_recall_hpc)
            recall_acc.append(mean_recall_acc)
            recall_output.append(mean_recall_output)
            recall_freezing.append(
                float(np.clip(100.0 * mean_recall_output[0] / freezing_fr_max, 0.0, 100.0))
            )

            restore_weights(model, saved_weights)
            model.rates = saved_rates
            model.rates_ctx = saved_acc_rates
            model.op_rate = saved_output_rate
            model.excitability = saved_hpc_ex
            model.excitability_ctx = saved_acc_ex

        all_hpc.append(hpc_history)
        all_acc.append(acc_history)
        all_output.append(output_history)
        all_input.append(input_history)
        all_ex_hpc.append(ex_hpc_history)
        all_ex_acc.append(ex_acc_history)
        all_last_hpc.append(last_hpc)
        all_last_acc.append(last_acc)
        all_recall_hpc.append(recall_hpc)
        all_recall_acc.append(recall_acc)
        all_recall_output.append(recall_output)
        all_recall_freezing.append(recall_freezing)
        all_nrem_masks.append(nrem_masks)
        all_rec_hpc.append(rec_hpc)
        all_rec_acc.append(rec_acc)
        all_hpc_acc.append(hpc_acc)
        all_acc_hpc.append(acc_hpc)
        all_hpc_output_w.append(hpc_output_w)
        all_acc_output_w.append(acc_output_w)
        final_model = model
        final_total_time = total_time

    arrays = {
        "FR_history.npy": all_hpc,
        "FR_history_ctx.npy": all_acc,
        "FR_history_op.npy": all_output,
        "input_history.npy": all_input,
        "EX_history.npy": all_ex_hpc,
        "EX_history_ctx.npy": all_ex_acc,
        "last_activity.npy": all_last_hpc,
        "last_activity_ctx.npy": all_last_acc,
        "recall_activity.npy": all_recall_hpc,
        "recall_activity_ctx.npy": all_recall_acc,
        "recall_activity_op.npy": all_recall_output,
        "recall_freezing_by_day.npy": all_recall_freezing,
        "nrem_reactivation_masks.npy": all_nrem_masks,
        "rec_weights.npy": all_rec_hpc,
        "rec_ctx_weights.npy": all_rec_acc,
        "mtl_ctx_weights.npy": all_hpc_acc,
        "ctx_mtl_weights.npy": all_acc_hpc,
        "mtl_op_weights.npy": all_hpc_output_w,
        "ctx_op_weights.npy": all_acc_output_w,
    }
    for filename, values in arrays.items():
        np.save(os.path.join(data_folder, filename), np.asarray(values))

    params = {
        "model_params": {
            "n_MTL": final_model.n_MTL,
            "n_CTX": final_model.n_CTX,
            "tau": final_model.tau,
            "dt": final_model.dt,
            "lr": final_model.lr,
            "lr_ctx": final_model.lr_ctx,
            "lr_ctx_late": final_model.lr_ctx_late,
            "lr_ctx_late_start_day": final_model.lr_ctx_late_start_day,
            "lr_mtl_ctx": final_model.lr_mtl_ctx,
            "lr_op_mtl": final_model.lr_op_mtl,
            "lr_op_ctx": final_model.lr_op_ctx,
        },
        "simulation_params": {
            "protocol": {
                "day_0": ["Learning", "REM", "NREM", "Recall"],
                "later_days": ["REM", "NREM", "Recall"],
            },
            "n": n,
            "n_inp": n_inp,
            "n_ctx": n_ctx,
            "FC_inp": fc_input,
            "E_fl": excitability_boost,
            "E_fl_ctx": acc_excitability_boost,
            "E_mod": tagged_acc_boost,
            "tau_IE": tagged_acc_tau,
            "threshold": 5.0,
            "N_off_days": n_days,
            "Nrep": learning_presentations,
            "t_off": learning_duration,
            "IR": learning_ir,
            "ID": rem_duration + nrem_duration,
            "rem_duration": rem_duration,
            "nrem_duration": nrem_duration,
            "nrem_reactivation_fraction": nrem_reactivation_fraction,
            "nrem_reactivated_neurons": n_reactivated,
            "recall_t": recall_duration,
            "N_recall": 1,
            "recall_days": list(range(n_days)),
            "NUM_SIM": num_sim,
            "start_seed": start_seed,
            "total_time": final_total_time,
            "dt": dt,
            "off_days": [0, 1, 2, 3, 7, 10],
            "t_series": [
                1000
                + learning_presentations * (learning_duration + learning_ir)
                + (day + 1) * (rem_duration + nrem_duration)
                for day in range(n_days)
            ],
            "freezing_fr_max": freezing_fr_max,
            "notes": (
                "Learning occurs on day 0 only. Every day then has REM, NREM, "
                "and a frozen-plasticity recall before the next offline day. "
                "Plasticity is enabled during day-0 learning and NREM only. "
                "NREM stimulates a random 75% of neuron/input channels."
            ),
        },
    }
    with open(os.path.join(data_folder, "all_params.json"), "w") as handle:
        json.dump(params, handle, indent=4)

    # The legacy PlotAll layout assumes learning presentations on every day,
    # which is not true for this protocol. Plot the dedicated recall measure
    # directly using its real day axis instead.
    os.makedirs(plot_folder, exist_ok=True)
    recall_freezing_array = np.asarray(all_recall_freezing)
    recall_mean = np.mean(recall_freezing_array, axis=0)
    recall_sem = np.std(recall_freezing_array, axis=0, ddof=1) / np.sqrt(num_sim)
    days = np.arange(n_days)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(days, recall_mean, marker="o", color="black")
    ax.fill_between(
        days,
        recall_mean - recall_sem,
        recall_mean + recall_sem,
        color="black",
        alpha=0.2,
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("Recall freezing (%)")
    ax.set_xticks(days)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_folder, "recall_freezing_by_day.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main_learning_offline()
