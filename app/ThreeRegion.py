import matplotlib.pyplot as plt
import json
from LoadNPLot2 import PlotAll3R
import numpy as np
from plotting_widget import *
from Utilities import average_firing_rates_with_active, ensamble_overlap
from tqdm import trange

start_seed = 5
np.random.seed(start_seed)


import numpy as np

def softplus(x):
    return np.log1p(np.exp(np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)


class threelayer_FF:
    """
    3-region rate model:
      HPC (was MTL) -> RSC -> ACC (was CTX) -> Output
    plus optional feedback ACC -> HPC (kept from your original as FB_ACC_HPC).
    """

    def __init__(
        self,
        n_inp,
        n_HPC,
        n_RSC,
        n_ACC,
        baseline_e_hpc,
        baseline_e_rsc,
        baseline_e_acc,
        tau=20.0,
        dt=1.0,
        act="relu",

        # HPC recurrent plasticity
        lr_hpc=1/800,
        decay_r_hpc=1/1000,

        # RSC recurrent plasticity
        lr_rsc=1/1200,
        decay_r_rsc=1e-5,

        # ACC recurrent plasticity
        lr_acc=1/1200,
        decay_r_acc=1e-5,

        # Feedforward plasticity
        lr_hpc_rsc=1/1000,
        decay_hpc_rsc=1e-5,
        lr_rsc_acc=1/1000,
        decay_rsc_acc=1e-5,
        lr_hpc_acc = 1/1000,
        decay_hpc_acc = 1e-5,

        # Feedback ACC->HPC plasticity (kept like your CTX->MTL)
        lr_acc_hpc=0,
        decay_acc_hpc=1,

        # Output learning
        lr_op=1/1000,

        # Inhibition params
        I0_hpc=1, I1_hpc=0.05, I2_hpc=0.001,
        I0_rsc=1, I1_rsc=0.05, I2_rsc=0.001,
        I0_acc=1, I1_acc=0.05, I2_acc=0.001,

        threshold=5,
        tag_threshold=2.0,
    ):
        self.n_HPC = n_HPC
        self.n_RSC = n_RSC
        self.n_ACC = n_ACC
        self.tau = tau
        self.dt = dt

        # Activation
        if act == "relu" or act == relu:
            self.act = relu
        elif act == "softplus" or act == softplus:
            self.act = softplus
        else:
            self.act = relu

        # Learning rates / decays
        self.lr_hpc = lr_hpc
        self.decay_r_hpc = decay_r_hpc

        self.lr_rsc = lr_rsc
        self.decay_r_rsc = decay_r_rsc

        self.lr_acc = lr_acc
        self.decay_r_acc = decay_r_acc

        self.lr_hpc_rsc = lr_hpc_rsc
        self.decay_hpc_rsc = decay_hpc_rsc

        self.lr_rsc_acc = lr_rsc_acc
        self.decay_rsc_acc = decay_rsc_acc
        
        self.lr_hpc_acc = lr_hpc_acc
        self.decay_hpc_acc = decay_hpc_acc

        self.lr_acc_hpc = lr_acc_hpc
        self.decay_acc_hpc = decay_acc_hpc

        self.lr_op = lr_op

        # Inhibition
        self.I0_hpc, self.I1_hpc, self.I2_hpc = I0_hpc, I1_hpc, I2_hpc
        self.I0_rsc, self.I1_rsc, self.I2_rsc = I0_rsc, I1_rsc, I2_rsc
        self.I0_acc, self.I1_acc, self.I2_acc = I0_acc, I1_acc, I2_acc

        # Thresholds
        self.plas_threshold = 0.2
        self.act_threshold_hpc = 0.0
        self.act_threshold_rsc = 0.0
        self.act_threshold_acc = 0.0
        self.threshold = threshold
        self.tag_threshold = tag_threshold

        # Excitability
        self.excitability_hpc = baseline_e_hpc.copy()
        self.excitability_rsc = baseline_e_rsc.copy()
        self.excitability_acc = baseline_e_acc.copy()

        # Gains / switches (mirroring your style)
        self.gain_hpc = 1.0
        self.gain_rsc = 1.0
        self.gain_acc = 1.0
        self.gain_HPC_RSC = 1.0
        self.gain_RSC_ACC = 1.0
        self.gain_HPC_ACC = 1.0

        self.on_hpc = 1.0
        self.on_rsc = 1.0
        self.on_acc = 1.0

        # Tagging
        self.tagged_HPC = np.zeros(n_HPC)
        self.tagged_RSC = np.zeros(n_RSC)
        self.tagged_ACC = np.zeros(n_ACC)

        # Weights
        self.input_w_hpc = np.abs(np.random.normal(0, 0.05, size=(n_inp, n_HPC)))

        self.rec_w_hpc = np.zeros((n_HPC, n_HPC))
        self.rec_w_rsc = np.zeros((n_RSC, n_RSC))
        self.rec_w_acc = np.zeros((n_ACC, n_ACC))

        # Feedforward
        self.FF_HPC_RSC = (1.0 / np.sqrt(n_RSC)) * np.abs(np.random.normal(0, 1.0, size=(n_RSC, n_HPC)))
        self.FF_RSC_ACC = (1.0 / np.sqrt(n_ACC)) * np.abs(np.random.normal(0, 1.0, size=(n_ACC, n_RSC)))
        self.FF_HPC_ACC = (1.0 / np.sqrt(n_ACC)) * np.abs(np.random.normal(0, 1.0, size=(n_ACC, n_HPC)))

        # Feedback (keep ACC -> HPC like your old CTX -> MTL)
        self.FB_ACC_HPC = 0.0 * (1.0 / np.sqrt(n_HPC)) * np.abs(np.random.normal(0, 0.05, size=(n_HPC, n_ACC)))

        # Output
        self.op_neuron = 1
        self.acc_op_w = 0.0 * np.abs(np.random.normal(0, 0.05, size=(self.op_neuron, n_ACC)))
        self.op_rate = np.zeros(self.op_neuron)

        # Rates
        self.rates_hpc = np.zeros(n_HPC)
        self.rates_rsc = np.zeros(n_RSC)
        self.rates_acc = np.zeros(n_ACC)

    def step(
        self,
        input_HPC,     # shape (n_HPC,) or (n_inp,) if you want to map via input_w_hpc; here assume already (n_HPC,)
        input_RSC,     # shape (n_RSC,)
        input_ACC,     # shape (n_ACC,)
        op_FR,         # scalar or (1,)
        op_signal=0.0,
        day_c=0,
        phase=None,
        ct=1
    ):
        # ---------- Compute region inputs ----------
        # HPC input includes recurrent + (optional) ACC feedback (subtract, like your old "- FB_CTX_MTL dot rates_acc")
        hpc_drive = input_HPC + self.rec_w_hpc.dot(self.rates_hpc) - self.FB_ACC_HPC.dot(self.rates_acc)

        # RSC input includes recurrent + feedforward from HPC
        rsc_drive = input_RSC + self.rec_w_rsc.dot(self.rates_rsc) + (self.gain_HPC_RSC * self.FF_HPC_RSC.dot(self.rates_hpc))

        # ACC input includes recurrent + feedforward from RSC and HPC
        acc_drive = input_ACC + self.rec_w_acc.dot(self.rates_acc) + (self.gain_RSC_ACC * self.FF_RSC_ACC.dot(self.rates_rsc)) + self.gain_HPC_ACC * self.FF_HPC_ACC.dot(self.rates_hpc)

        # Output reads out from ACC (same idea as your ctx_op_w.dot(rates_acc))
        input_op = op_FR + self.acc_op_w.dot(self.rates_acc)

        # ---------- Inhibition ----------
        I_hpc = self.I0_hpc + self.I1_hpc * np.sum(self.rates_hpc) + self.I2_hpc * np.sum(self.rates_hpc**2)
        I_rsc = self.I0_rsc + self.I1_rsc * np.sum(self.rates_rsc) + self.I2_rsc * np.sum(self.rates_rsc**2)
        I_acc = self.I0_acc + self.I1_acc * np.sum(self.rates_acc) + self.I2_acc * np.sum(self.rates_acc**2)

        hpc_current = hpc_drive - I_hpc
        rsc_current = rsc_drive - I_rsc
        acc_current = acc_drive - I_acc

        # ---------- Tagging ----------
        self.tagged_HPC = self.tagged_HPC + (((self.rates_hpc > self.tag_threshold).astype(float) *
                                              (self.tagged_HPC == 0).astype(float)) * day_c)
        self.tagged_RSC = self.tagged_RSC + (((self.rates_rsc > self.tag_threshold).astype(float) *
                                              (self.tagged_RSC == 0).astype(float)) * day_c)
        self.tagged_ACC = self.tagged_ACC + (((self.rates_acc > self.tag_threshold).astype(float) *
                                              (self.tagged_ACC == 0).astype(float)) * day_c)

        # ---------- Rate dynamics ----------
        dr_hpc = (-self.rates_hpc + self.act(self.gain_hpc * (self.excitability_hpc + hpc_current + self.act_threshold_hpc))) / self.tau
        dr_rsc = (-self.rates_rsc + self.act(self.gain_rsc * (self.excitability_rsc + rsc_current + self.act_threshold_rsc))) / self.tau
        dr_acc = (-self.rates_acc + self.act(self.gain_acc * (self.excitability_acc + acc_current + self.act_threshold_acc))) / self.tau
        dr_op  = (-self.op_rate    + self.act(input_op)) / self.tau

        self.rates_hpc += dr_hpc * self.dt
        self.rates_rsc += dr_rsc * self.dt
        self.rates_acc += dr_acc * self.dt
        self.op_rate   += dr_op  * self.dt

        # Clip rates
        self.rates_hpc = np.clip(self.rates_hpc, 0.0, 10.0)
        self.rates_rsc = np.clip(self.rates_rsc, 0.0, 10.0)
        self.rates_acc = np.clip(self.rates_acc, 0.0, 10.0)

        # ---------- Plasticity ----------
        # HPC recurrent
        post_hpc = self.rates_hpc > self.plas_threshold
        heb_hpc = self.on_hpc * self.lr_hpc * np.outer(self.rates_hpc * post_hpc, self.rates_hpc * post_hpc) * self.dt
        dec_hpc = self.decay_r_hpc * self.rec_w_hpc * self.dt
        self.rec_w_hpc += (heb_hpc - dec_hpc)

        # RSC recurrent
        post_rsc = self.rates_rsc > self.plas_threshold
        heb_rsc = self.on_rsc * self.lr_rsc * np.outer(self.rates_rsc * post_rsc, self.rates_rsc * post_rsc) * self.dt
        dec_rsc = self.decay_r_rsc * self.rec_w_rsc * self.dt
        self.rec_w_rsc += (heb_rsc - dec_rsc)

        # ACC recurrent
        post_acc = self.rates_acc > self.plas_threshold
        heb_acc = self.on_acc * self.lr_acc * np.outer(self.rates_acc * post_acc, self.rates_acc * post_acc) * self.dt
        dec_acc = self.decay_r_acc * self.rec_w_acc * self.dt
        self.rec_w_acc += (heb_acc - dec_acc)

        # FF: HPC -> RSC
        heb_hpc_rsc = self.on_rsc * self.lr_hpc_rsc * np.outer(self.rates_rsc * post_rsc, self.rates_hpc * post_hpc) * self.dt
        dec_hpc_rsc = self.decay_hpc_rsc * self.FF_HPC_RSC * self.dt
        self.FF_HPC_RSC += (heb_hpc_rsc - dec_hpc_rsc)

        # FF: RSC -> ACC
        heb_rsc_acc = self.on_acc * self.lr_rsc_acc * np.outer(self.rates_acc * post_acc, self.rates_rsc * post_rsc) * self.dt
        dec_rsc_acc = self.decay_rsc_acc * self.FF_RSC_ACC * self.dt
        self.FF_RSC_ACC += (heb_rsc_acc - dec_rsc_acc)
        
        # FF: HPC -> ACC (shortcut)
        heb_hpc_acc = self.on_acc * self.lr_hpc_acc * np.outer(self.rates_acc * post_acc,self.rates_hpc * post_hpc) * self.dt
        dec_hpc_acc = self.decay_hpc_acc * self.FF_HPC_ACC * self.dt
        self.FF_HPC_ACC += (heb_hpc_acc - dec_hpc_acc)

        # FB: ACC -> HPC (kept analogous to your old CTX->MTL)
        heb_acc_hpc = self.on_hpc * self.lr_acc_hpc * np.outer(self.rates_hpc * post_hpc, self.rates_acc * post_acc) * self.dt
        dec_acc_hpc = self.decay_acc_hpc * self.FB_ACC_HPC * self.dt
        self.FB_ACC_HPC += (heb_acc_hpc - dec_acc_hpc)

        # Output: ACC -> op
        heb_acc_op = self.lr_op * op_signal * np.outer(self.op_rate, self.rates_acc * post_acc) * self.dt
        dec_acc_op = 1e-8 * self.acc_op_w * self.dt
        self.acc_op_w += (heb_acc_op - dec_acc_op)

        # Normalize output weights (same spirit as yours)
        sum_w = np.sum(self.acc_op_w)
        if sum_w >= 2:
            self.acc_op_w *= (2.0 / sum_w)
        self.acc_op_w = np.clip(self.acc_op_w, 0.0, None)

        # Clip weights
        self.rec_w_hpc = np.clip(self.rec_w_hpc, 0.0, 1.0)
        self.rec_w_rsc = np.clip(self.rec_w_rsc, 0.0, 1.0)
        self.rec_w_acc = np.clip(self.rec_w_acc, 0.0, 1.0)
        self.FF_HPC_RSC = np.clip(self.FF_HPC_RSC, 0.0, 1.0)
        self.FF_RSC_ACC = np.clip(self.FF_RSC_ACC, 0.0, 1.0)
        self.FF_HPC_ACC = np.clip(self.FF_HPC_ACC, 0.0, 1.0)
        # self.FB_ACC_HPC = np.clip(self.FB_ACC_HPC, 0.0, 1.0)

        return self.rates_hpc, self.rates_rsc, self.rates_acc, self.op_rate
    
    def _normalize_input_outgoing(self, target_sum=None, eps=1e-12):
        """Normalize columns so that for each input feature, the outgoing weights sum to target_sum."""
        if target_sum is None:
            target_sum = self.target_out_sum
        col_sums = self.rec_w_hpc.sum(axis=1, keepdims=True)
        scale = np.where(col_sums > eps, target_sum / col_sums, np.ones_like(col_sums))
        self.rec_w = self.rec_w * scale


# Main simulation code
# FR_history = []
# rec_HPC_weights = []
rec_RSC_weights = []
rec_RSC_weights = []
# ff_weights = []
# last_activity = []
# input_history = []


def main():
    N_off_days = 11
    N_neurons_per_day = 20
    n = 10 + (N_off_days) * N_neurons_per_day
    n_inp = n
    n_hpc = n
    n_rsc = n
    n_acc = n

    E_fl_hpc = 2.5
    E_fl_acc = 2.5
    E_fl_rsc = 2.5
    E_mod = 2.5
    max_e = 10
    E_ref = 0.7
    threshold = 5.0
    off_set = 0

    IP_plasticity_limit = 7
    condition = ""
 
    FC_inp = 15
    FC_d0_inp = 15
    FC_inp_acc = 15
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

    FR_HPC_history_all = []
    FR_RSC_history_all = []
    FR_ACC_history_all = []
    FR_op_history_all = []

    last_activity_HPC_all = []
    last_activity_RSC_all = []
    last_activity_ACC_all = []

    EX_HPC_history_all = []
    EX_RSC_history_all = []
    EX_ACC_history_all = []

    rec_HPC_weights_all = []
    rec_ACC_weights_all = []
    rec_RSC_weights_all = []

    HPC_ACC_weights_all = []
    ACC_HPC_weights_all = []
    HPC_RSC_weights_all = []
    RSC_ACC_weights_all = []

    ACC_OP_weights_all = []
    HPC_OP_weights_all = []

    dob_HPC = []
    dob_ACC = []
    off_days = [0, 1, 2, 3, 7, 10]
    if not dob_ACC == []:
        condition += "3R_ACCBlock{}".format(dob_ACC)
    if not dob_HPC == []:
        condition += "3R_HPCBlock{}".format(dob_HPC)
    else:
        condition +="3R_CNT"
    if E_mod == 0.0:
        sim_name = "{}_fast_drift_wo_IP_lowI".format(condition)
    else:
        sim_name = "{}_fast_drift_with_limited{}_IP_lowI".format(condition, IP_plasticity_limit)

    notes = "2 region model with slow drift due to low excitability boosts in both regions with intrinsic plasticity. ACC neurons that are part of the FC engram get an extra boost in excitability during off days."

    for i in trange(NUM_SIM):
        t_series = []
        total_time = 0
        np.random.seed(start_seed + i)
    
        base_E = np.abs(np.random.normal(mu_ex, sigma_ex, size=(n,)))
        base_e_acc = np.abs(np.random.normal(mu_ex, sigma_ex, size=(n_acc,)))
        base_e_rsc = np.abs(np.random.normal(mu_ex, sigma_ex, size=(n_rsc,)))
    
        nn = threelayer_FF(
            n_inp=n_inp,
            n_HPC=n_hpc,
            n_RSC=n_rsc,
            n_ACC=n_acc,
            baseline_e_hpc=base_E.copy(),
            baseline_e_rsc=base_e_rsc.copy(),
            baseline_e_acc=base_e_acc.copy(),
            tau=20.0,
            dt=dt,
            act="relu",

            lr_hpc=1./3000, decay_r_hpc=1./3500,
            lr_rsc=2e-6,    decay_r_rsc=0.,
            lr_acc=2e-6,    decay_r_acc=0.,

            lr_hpc_rsc=1/3000., decay_hpc_rsc=1e-6,
            lr_rsc_acc=1/3000., decay_rsc_acc=1e-6,

            lr_acc_hpc=0., decay_acc_hpc=0.,  # matches your old ctx_mtl off if you want
            lr_op=1e-3,

            I0_hpc=5, I1_hpc=0.7, I2_hpc=0.04,
            I0_rsc=2., I1_rsc=0.6, I2_rsc=0.04,
            I0_acc=2., I1_acc=0.6, I2_acc=0.04,
        )
    
        input_history = []
    
        FR_HPC_history = []
        FR_RSC_history = []
        FR_ACC_history = []
        FR_op_history = []
    
        last_activity_hpc = []
        last_activity_rsc = []
        last_activity_acc = []
    
        EX_HPC_history = []
        EX_ACC_history = []
        EX_RSC_history = []
    
        rec_HPC_weights = []
        rec_RSC_weights = []
        rec_ACC_weights = []
    
        HPC_ACC_weights = []
        ACC_HPC_weights = []
        HPC_RSC_weights = []
        RSC_ACC_weights = []
    
        rep_Activity = []
        high_threshold = 5
    
        HPC_OP_weights = []
        ACC_OP_weights = []
    
        nn.excitability_hpc = base_E.copy()
        day = 0
    
        nn.excitability_hpc[off_set+(day)*N_neurons_per_day:off_set+(day)*N_neurons_per_day+N_neurons_per_day] += E_fl_hpc
        nn.excitability_rsc = base_e_rsc.copy()
        nn.excitability_rsc[(off_set+(day)*N_neurons_per_day):(off_set+(day)*N_neurons_per_day+N_neurons_per_day)] += E_fl_rsc
        nn.excitability_acc = base_e_acc.copy()
        nn.excitability_acc[(off_set+(day)*N_neurons_per_day):(off_set+(day)*N_neurons_per_day+N_neurons_per_day)] += E_fl_acc

        for t in range(ID):
            FR_HPC, FR_RSC, FR_ACC, FR_op = nn.step(zero_input, zero_input,zero_input, 0, 0, day_c=-1, phase="Burnoff", ct=total_time+t)
            FR_HPC_history.append(FR_HPC)
            FR_RSC_history.append(FR_RSC)
            FR_ACC_history.append(FR_ACC)
            FR_op_history.append(FR_op)
            EX_HPC_history.append(nn.excitability_hpc.copy())
            EX_ACC_history.append(nn.excitability_acc.copy())
            input_history.append(zero_input.copy())
    
        total_time += ID
        t_series.append(total_time)
    
        for day in range(N_off_days):
            day_activity_hpc = []
            day_activity_rsc = []
            day_activity_acc = []
            nn.excitability_hpc = base_E.copy()
            nn.excitability_rsc = base_e_rsc.copy()
            nn.excitability_acc = base_e_acc.copy()
            nn.excitability_hpc[off_set+(day)*N_neurons_per_day:off_set+(day)*N_neurons_per_day+N_neurons_per_day] += E_fl_hpc
            new_neurons = range(off_set + day*N_neurons_per_day, off_set + day*N_neurons_per_day + N_neurons_per_day, 1)
            nn.excitability_rsc = base_e_rsc.copy()
            nn.excitability_rsc[(off_set+(day)*N_neurons_per_day):(off_set+(day)*N_neurons_per_day+N_neurons_per_day)] += E_fl_rsc

            if day <= IP_plasticity_limit:
                FC_acc_active_neurons = np.where(nn.tagged_ACC == 1.)[0]
                FC_HPC_active_neurons = np.where(nn.tagged_HPC == 1.)[0]
                nn.excitability_acc[FC_acc_active_neurons] += E_mod
                for n1 in new_neurons:
                    if n1 not in FC_acc_active_neurons:
                        nn.excitability_acc[n1] += E_fl_acc
            else:
                nn.excitability_acc[new_neurons] += E_fl_acc
        
            ctx_inp = FC_inp_acc
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
                if day in dob_ACC:
                    nn.on_acc = 0.0
                else:
                    nn.on_acc = 1.0
            
                if day in dob_HPC:
                    nn.on = 0.0
                else:
                    nn.on = 1.0

                nn.FF_plas = 1.0
                nn.gain_FF = 1.0
            
                for t in range(t_off):
                    FR_HPC, FR_RSC, FR_ACC, FR_op = nn.step(input,zero_input, ctx_inp, op_inp, op_learning, day_c=day+1, phase=phase, ct=total_time+t)
                    FR_HPC_history.append(FR_HPC)
                    FR_RSC_history.append(FR_RSC)
                    FR_ACC_history.append(FR_ACC)
                    FR_op_history.append(FR_op)
                    EX_HPC_history.append(nn.excitability_hpc.copy())
                    EX_RSC_history.append(nn.excitability_rsc.copy())  
                    EX_ACC_history.append(nn.excitability_acc.copy())
                    input_history.append(input.copy())
            
                day_activity_hpc.append(np.mean(FR_HPC_history[-t_off//2:], axis=0))
                day_activity_rsc.append(np.mean(FR_RSC_history[-t_off//2:], axis=0))
                day_activity_acc.append(np.mean(FR_ACC_history[-t_off//2:], axis=0))
                total_time += t_off
                t_series.append(total_time)
            
                for t in range(IR):
                    FR_HPC, FR_RSC, FR_ACC, FR_op = nn.step(zero_input,zero_input, zero_input, 0, 0, day_c=day+1, phase="IR", ct=total_time)
                    FR_HPC_history.append(FR_HPC)
                    FR_RSC_history.append(FR_RSC)
                    FR_ACC_history.append(FR_ACC)
                    FR_op_history.append(FR_op)
                    EX_HPC_history.append(nn.excitability_hpc.copy())
                    EX_RSC_history.append(nn.excitability_rsc.copy())
                    EX_ACC_history.append(nn.excitability_acc.copy())
                    input_history.append(zero_input.copy())
            
                total_time += IR
                if rep != Nrep - 1:
                    t_series.append(total_time)
        
            nn.FF_plas = 0.0
            nn.gain_FF = 0.0

            rec_HPC_weights.append(nn.rec_w_hpc.copy())
            rec_RSC_weights.append(nn.rec_w_rsc.copy())
            rec_ACC_weights.append(nn.rec_w_acc.copy())
        
            HPC_ACC_weights.append(nn.FF_HPC_ACC.copy())
            HPC_RSC_weights.append(nn.FF_HPC_RSC.copy())
            RSC_ACC_weights.append(nn.FF_RSC_ACC.copy())
            ACC_HPC_weights.append(nn.FB_ACC_HPC.copy())
        
            # HPC_OP_weights.append(nn.mtl_op_w.copy())
            ACC_OP_weights.append(nn.acc_op_w.copy())
        
            # rep_Activity.append(day_activity)
            last_activity_hpc.append(day_activity_hpc[-1])
            last_activity_rsc.append(day_activity_rsc[-1])
            last_activity_acc.append(day_activity_acc[-1])
        
            for t in range(ID):
                FR_HPC, FR_RSC, FR_ACC, FR_op = nn.step(zero_input, zero_input,zero_input, 0, day_c=day+1, phase="REM", ct=total_time+t)
                FR_HPC_history.append(FR_HPC)
                FR_RSC_history.append(FR_RSC)
                FR_ACC_history.append(FR_ACC)
                FR_op_history.append(FR_op)
                EX_HPC_history.append(nn.excitability_hpc.copy())
                EX_RSC_history.append(nn.excitability_rsc.copy())
                EX_ACC_history.append(nn.excitability_acc.copy())
                input_history.append(zero_input.copy()) 
            total_time += ID
            t_series.append(total_time)
    
        input_history_all.append(input_history)
    
        FR_HPC_history_all.append(FR_HPC_history)
        FR_RSC_history_all.append(FR_RSC_history)
        FR_ACC_history_all.append(FR_ACC_history)
        FR_op_history_all.append(FR_op_history)
    
        last_activity_HPC_all.append(last_activity_hpc)
        last_activity_RSC_all.append(last_activity_rsc)
        last_activity_ACC_all.append(last_activity_acc)
    
        EX_HPC_history_all.append(EX_HPC_history)
        EX_RSC_history_all.append(EX_RSC_history)
        EX_ACC_history_all.append(EX_ACC_history)
    
        rec_HPC_weights_all.append(rec_HPC_weights)
        rec_ACC_weights_all.append(rec_ACC_weights)
        rec_RSC_weights_all.append(rec_RSC_weights)
    
        HPC_ACC_weights_all.append(HPC_ACC_weights)
        HPC_RSC_weights_all.append(HPC_RSC_weights)
        RSC_ACC_weights_all.append(RSC_ACC_weights)
        ACC_HPC_weights_all.append(ACC_HPC_weights)
    
        # HPC_OP_weights_all.append(HPC_OP_weights)
        ACC_OP_weights_all.append(ACC_OP_weights)

    FR_HPC_history_all = np.stack(FR_HPC_history_all)
    FR_ACC_history_all = np.stack(FR_ACC_history_all)
    FR_RSC_history_all = np.stack(FR_RSC_history_all)
    FR_op_history_all = np.stack(FR_op_history_all)

    input_history = np.stack(input_history)

    EX_HPC_history_all = np.stack(EX_HPC_history_all)
    EX_RSC_history_all = np.stack(EX_RSC_history_all)
    EX_ACC_history_all = np.stack(EX_ACC_history_all)

    last_activity_HPC_all = np.stack(last_activity_HPC_all)
    last_activity_RSC_all = np.stack(last_activity_RSC_all)
    last_activity_ACC_all = np.stack(last_activity_ACC_all)

    rec_HPC_weights_all = np.stack(rec_HPC_weights_all)
    rec_RSC_weights_all = np.stack(rec_RSC_weights_all)
    rec_ACC_weights_all = np.stack(rec_ACC_weights_all)

    HPC_RSC_weights_all = np.stack(HPC_RSC_weights_all)
    RSC_ACC_weights_all = np.stack(RSC_ACC_weights_all)
    HPC_ACC_weights_all = np.stack(HPC_ACC_weights_all)
    ACC_HPC_weights_all = np.stack(ACC_HPC_weights_all)
    ACC_OP_weights_all = np.stack(ACC_OP_weights_all)

    # HPC_OP_weights_all = np.stack(HPC_OP_weights_all)

    op_data_folder = "../data/{}".format(sim_name)
    op_plot_folder = "../plots/{}".format(sim_name)

    import os
    os.makedirs(op_data_folder, exist_ok=True)
    np.save("{}/FR_HPC_history_all.npy".format(op_data_folder), FR_HPC_history_all)
    np.save("{}/FR_RSC_history_all.npy".format(op_data_folder), FR_RSC_history_all)
    np.save("{}/FR_ACC_history_all.npy".format(op_data_folder), FR_ACC_history_all)
    np.save("{}/FR_op_history_all.npy".format(op_data_folder), FR_op_history_all)

    np.save("{}/EX_HPC_history_all.npy".format(op_data_folder), EX_HPC_history_all)
    np.save("{}/EX_RSC_history_all.npy".format(op_data_folder), EX_RSC_history_all)
    np.save("{}/EX_ACC_history_all.npy".format(op_data_folder), EX_ACC_history_all)

    np.save("{}/last_activity_HPC_all.npy".format(op_data_folder), last_activity_HPC_all)
    np.save("{}/last_activity_RSC_all.npy".format(op_data_folder), last_activity_RSC_all)
    np.save("{}/last_activity_ACC_all.npy".format(op_data_folder), last_activity_ACC_all)

    np.save("{}/input_history_all.npy".format(op_data_folder), input_history_all)
    np.save("{}/rec_HPC_weights_all.npy".format(op_data_folder), rec_HPC_weights_all)
    np.save("{}/rec_RSC_weights_all.npy".format(op_data_folder), rec_RSC_weights_all)
    np.save("{}/rec_ACC_weights_all.npy".format(op_data_folder), rec_ACC_weights_all)
    np.save("{}/HPC_RSC_weights_all.npy".format(op_data_folder), HPC_RSC_weights_all)
    np.save("{}/RSC_ACC_weights_all.npy".format(op_data_folder), RSC_ACC_weights_all)   
    np.save("{}/HPC_ACC_weights_all.npy".format(op_data_folder), HPC_ACC_weights_all)

    # np.save("{}/HPC_OP_weights.npy".format(op_data_folder), HPC_OP_weights_all)
    np.save("{}/ACC_OP_weights_all.npy".format(op_data_folder), ACC_OP_weights_all)
    np.save("{}/ACC_HPC_weights_all.npy".format(op_data_folder), ACC_HPC_weights_all)

    sim_params = {
        "n": [n_hpc,n_rsc,n_acc],
        "n_inp": n_inp,
        "FC_inp": FC_inp,
        "E_fl": [E_fl_hpc, E_fl_rsc, E_fl_acc],
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
            "n_MTL": nn.n_HPC,
            "tau": nn.tau,
            "dt": nn.dt,
            "lr_hpc": nn.lr_hpc,
            "decay_r_hpc": nn.decay_r_hpc,
            "lr_rsc": nn.lr_rsc,
            "decay_r_rsc": nn.decay_r_rsc,
            "lr_acc": nn.lr_acc,
            "decay_r_acc": nn.decay_r_acc, 
            "lr_hpc_rsc": nn.lr_hpc_rsc,
            "decay_hpc_rsc": nn.decay_hpc_rsc,
            "lr_rsc_acc": nn.lr_rsc_acc,
            "decay_rsc_acc": nn.decay_rsc_acc,
            "lr_hpc_acc": nn.lr_hpc_acc,
            "decay_hpc_acc": nn.decay_hpc_acc,
            "lr_acc_hpc": nn.lr_acc_hpc,
            "decay_acc_hpc": nn.decay_acc_hpc,
            "lr_op": nn.lr_op,
            "Inhib_HPC": [nn.I0_hpc,nn.I1_hpc,nn.I2_hpc],
            "Inhib_RSC": [nn.I0_rsc,nn.I1_rsc,nn.I2_rsc],
            "Inhib_ACC": [nn.I0_acc,nn.I1_acc,nn.I2_acc],
            "mu_ex": mu_ex,
            "sigma_ex": sigma_ex
        },
        "simulation_params": sim_params
    }

    filename = "{}/all_params.json".format(op_data_folder)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    breakpoint()
    PlotAll3R(input_data_folder=op_data_folder, op_plot_folder=op_plot_folder)


if __name__ == "__main__":
    main()
