import matplotlib.pyplot as plt
import json
import os
import numpy as np
from LoadNPLot2 import PlotAll
from plotting_widget import *
from Utilities import average_firing_rates_with_active,ensamble_overlap
from tqdm import trange
start_seed = 0
np.random.seed(start_seed)


def softplus(x):
    """Numerically stable NumPy implementation of softplus."""
    return np.log1p(np.exp(np.clip(x, -500, 500)))


def relu(x):
    return np.maximum(0, x)


def tanh_window(t, duration, ramp_width=6.0, ramp_time=10.0):
    """Smoothly ramp up and down within one stimulation epoch."""
    ramp_up = 0.5 * (1.0 + np.tanh((t - ramp_time) / ramp_width))
    ramp_down = 0.5 * (1.0 + np.tanh((duration - 1 - t - ramp_time) / ramp_width))
    return ramp_up * ramp_down


class twolayer_FF:
    def __init__(self, n_inp, n_MTL,n_CTX,baseline_e,base_e_ctx,tau=20.0, dt=1.0, act=softplus, 
                 lr_hpc=1/800, decay_r_hpc=1/1000, 
                 lr_ctx = 1/1200, decay_r_ctx = 1e-5,
                 lr_op_MTL = 1/1000, lr_op_CTX = 1/1000,
                 I0=1, I1=0.05, I2=0.001,
                 I0_ctx=1, I1_ctx=0.05, I2_ctx=0.001,
                 threshold=5.):
        self.n_MTL = n_MTL
        self.n_CTX = n_CTX
        self.tau = tau  # rate constant 
        self.dt = dt  # discretization step
        if act == "relu" or act == relu:
            self.act = relu
        elif act == "softplus" or act == softplus:
            self.act = softplus
        elif callable(act):
            self.act = act
        else:
            self.act = relu
        self.lr_hpc = lr_hpc  # learning rate for synaptic weights
        self.decay_r_hpc = decay_r_hpc  # decay rate for synaptic weights
        self.lr_ctx = lr_ctx  # learning rate for synaptic weights
        self.decay_r_ctx = decay_r_ctx
        self.lr_op_MTL = lr_op_MTL
        self.lr_op_CTX = lr_op_CTX
        self.I0 = I0
        self.I1 = I1
        self.I2 = I2

        self.I0_ctx = I0_ctx
        self.I1_ctx = I1_ctx
        self.I2_ctx = I2_ctx

        self.excitability = baseline_e
        self.excitability_ctx = base_e_ctx
        # self.plas_threshold = 0
        self.threshold = threshold
        self.FF_MTL_CTX = 0.
        self.FB_CTX_MTL = 0.
        self.gain_ctx = 1.
        self.gain_hpc = 1.
        self.op_neuron = 1
        self.on = 1.0  # plasticity on/off
        self.on_ctx = 1.0
        # Initialize random input weights
        self.input_w = np.abs(np.random.normal(0,0.05,size=(n_inp,n_MTL)))
        # self.input_w = np.clip(self.input_w, 0.0, 0.1)
        self.rec_w = np.zeros((n_MTL, n_MTL))
        self.rec_w_ctx = np.zeros((self.n_CTX, self.n_CTX))
        
        self.mtl_op_w = np.abs(np.random.normal(0,0.05,size=(self.op_neuron,n_MTL)))
        self.ctx_op_w = np.abs(np.random.normal(0,0.05,size=(self.op_neuron,n_CTX)))
        self.op_lr_threshold = 2.
        # Zero initial rate state
        self.rates_ctx = np.zeros(self.n_CTX)
        self.rates = np.zeros(n_MTL)
        self.op_rate = np.zeros(self.op_neuron)
        # self.
    def TurnOFF_FB(self):
        self.FB_CTX_MTL = 0.0
    def TurnON_FB(self):
        self.FB_CTX_MTL = 0.0
    def TurnOFF_FF(self):
        self.FF_MTL_CTX = 0.0
    def TurnON_FF(self):
        self.FF_MTL_CTX = 1.0

    def step(self, input_FR, op_signal = 0,op_inp = 0):
        """
        Perform one timestep of rate dynamics:
        input_vector: shape [n_MTL]
        """
        # calculating the input to the RNN
        self.rates *= (self.rates > 1e-5).astype(float)  # Ensure rates are non-negative
        self.rates_ctx *= (self.rates_ctx > 1e-5).astype(float)
        input_vector = input_FR + self.rec_w @ self.rates - self.FB_CTX_MTL * self.rates_ctx
        input_CTX = input_FR + self.rec_w_ctx @ self.rates_ctx + self.FF_MTL_CTX * self.rates
        input_op = self.mtl_op_w @ self.rates + self.ctx_op_w @ self.rates_ctx + op_inp

        # blanket inhibition to the RNN
        I_inhib = self.I0 + self.I1 * np.sum(self.rates) + self.I2 * np.sum(self.rates**2)
        I_inhib_ctx = self.I0_ctx + self.I1_ctx * np.sum(self.rates_ctx) + self.I2_ctx * np.sum(self.rates_ctx**2)

        # total input to the RNN
        input_current =  input_vector -  I_inhib 
        input_ctx = input_CTX - I_inhib_ctx

        # rate change as the nonlinear ODE
        dr_dt = (-self.rates +   self.act((self.excitability + input_current ))) / self.tau
        dr_ctx_dt = (-self.rates_ctx + self.act((self.excitability_ctx + input_ctx ))) / self.tau

        dr_op_dt = (-self.op_rate + self.act(input_op)) / self.tau

        
        self.rates += (dr_dt * self.dt)
        self.rates_ctx += (dr_ctx_dt * self.dt)
        self.op_rate += (dr_op_dt*self.dt)

        # hebbian plasticity in RNN weights
        hebbian_dw = self.on * self.lr_hpc * np.outer(self.rates, self.rates) * self.dt
        decay = self.decay_r_hpc * self.rec_w * self.dt


        # hebbian plasticity in input weights
        self.rec_w += (hebbian_dw - decay)

        hebb_mtl_op = self.lr_op_MTL * op_signal * np.outer(self.op_rate, self.rates) * self.dt
        self.mtl_op_w += hebb_mtl_op
        
        hebbian_dw_ctx = self.on_ctx * self.lr_ctx * np.outer(self.rates_ctx, self.rates_ctx) * self.dt
        decay_ctx = self.decay_r_ctx * self.rec_w_ctx * self.dt
        self.rec_w_ctx += (hebbian_dw_ctx - decay_ctx)
        
        hebb_ctx_op = self.lr_op_CTX * op_signal * np.outer(self.op_rate, self.rates_ctx > self.op_lr_threshold) * self.dt

        self.ctx_op_w += hebb_ctx_op
        sum_w = (np.sum(self.ctx_op_w))
        self.ctx_op_w /= sum_w
        self.ctx_op_w = np.clip(self.ctx_op_w, 0, None)
        sum_w = (np.sum(self.mtl_op_w))
        self.mtl_op_w /= sum_w
        self.mtl_op_w = np.clip(self.mtl_op_w, 0, None)

        self.rates = np.clip(self.rates, 0.0, 15.)
        self.rates_ctx = np.clip(self.rates_ctx, 0.0, 15.)

        # self.rates = np.clip(self.rates, 0.0, 15)  # Ensure rates are non-negative
        self.rec_w = np.clip(self.rec_w, 0.0, 1.0)  # Ensure weights are non-negative
        self.rec_w_ctx = np.clip(self.rec_w_ctx, 0.0, 1.0)
        
        
        return self.rates.copy(),self.rates_ctx.copy(), self.op_rate.copy()


def main():
    FR_history = []
    EX_history = []
    rec_weights = []
    ff_weights = []
    last_activity = []
    input_history = []



    N_off_days = 11
    n_neurons_per_day = 20
    n = 10 + (N_off_days) * n_neurons_per_day #10 default + 20 neurons per off day
    n_inp = n 
    n_ctx = n 
    E_fl = [1.8] #np.arange(0,4,0.4)
    E_fl_ctx = [1.8]#np.arange(0,4,0.4)
    max_e = 10
    E_ref = 0.7
    threshold = 5
    off_set = 0



    # base_E[:off_set] += 2
    sim_name = "CNT_fast_drift_FF"
    notes = "trying with Feedback and feedforward connection of equal strength"
    FC_inp = 13
    input = FC_inp*np.ones(n_inp)
    zero_input = np.zeros(n_inp)
    mu_ex = 0
    sigma_ex = 1.0
    mean_DR = []
    mean_DR_ctx = []
    off_days = [0, 1, 2, 3, 7, 10]
    for x1,x2 in zip(E_fl,E_fl_ctx):
        ID = 1000
        dt = 1
        NUM_SIM = 10
        t_off = 100
        input_ramp_time = 10.0
        input_ramp_width = 6.0
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
        dob = []
        t_encoding = 1000

        for i in trange(NUM_SIM):
            t_series = []
            total_time = 0
            np.random.seed(start_seed + i)
            base_E = np.abs(np.random.normal(mu_ex,sigma_ex,size=(n,)))
            base_e_ctx = np.abs(np.random.normal(mu_ex,sigma_ex,size=(n_ctx,)))
            nn = twolayer_FF(n_inp=n_inp, n_MTL=n,n_CTX=n_ctx, baseline_e = base_E.copy(),
                            base_e_ctx=base_e_ctx.copy(), tau=20.0, dt=dt, act=relu,
                            lr_hpc=1/800, decay_r_hpc=1/1000, 
                            lr_ctx = 1e-5,decay_r_ctx=1e-8,
                            lr_op_MTL = 1e-3, lr_op_CTX = 1e-3,
                            I0=9., I1=0.5, I2=0.05,
                            I0_ctx=9., I1_ctx=0.5, I2_ctx=0.05)
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
            nn.excitability = base_E.copy()
            day = 0
            nn.excitability[off_set+(day)*n_neurons_per_day:off_set+(day)*n_neurons_per_day+n_neurons_per_day] += x1
            # nn.excitability = np.clip(nn.excitability,0,max_e)
            nn.excitability_ctx  = base_e_ctx.copy()
            nn.excitability_ctx[(off_set+(day)*n_neurons_per_day):(off_set+(day)*n_neurons_per_day+n_neurons_per_day)] += x2
        
            for t in range(ID):
                next_FR,FR_ctx,FR_op = nn.step(zero_input,0,0)
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
                nn.excitability = base_E.copy()
                nn.excitability[off_set+(day)*n_neurons_per_day:off_set+(day)*n_neurons_per_day+n_neurons_per_day] += x1
                # nn.excitability = np.clip(nn.excitability,0,max_e)
                nn.excitability_ctx  = base_e_ctx.copy()
                nn.excitability_ctx[(off_set+(day)*n_neurons_per_day):(off_set+(day)*n_neurons_per_day + n_neurons_per_day)] += x2
                inp_to_network = input                
    
                for rep in range(Nrep):
                    nn.TurnON_FB()
                    nn.TurnON_FF()
                
                    # if day in dob:
                    #     nn.gain_hpc = 0.0
                    #     # nn.TurnOFF_FB()
                    # else:
                    #     nn.gain_hpc  = 1.0
                        # nn.TurnON_FB()
                    if day == 0:
                        op_learning = 1.0
                        op_inp = 1.0
                    else:
                        op_learning = 0.0
                        op_inp = 0.0
                        # nn.on = 0.3
                    for t in range(t_off):
                        ramp = tanh_window(t, t_off, ramp_width=input_ramp_width, ramp_time=input_ramp_time)
                        ramped_input = ramp * inp_to_network
                        ramped_op_inp = ramp * op_inp
                        next_FR,FR_ctx,FR_op = nn.step(ramped_input,op_learning,ramped_op_inp)
                        FR_history.append(next_FR)
                        FR_history_ctx.append(FR_ctx)
                        FR_op_history.append(FR_op)
                        EX_history.append(nn.excitability.copy())
                        EX_history_ctx.append(nn.excitability_ctx.copy())
                        input_history.append(ramped_input.copy())
                    day_activity.append(np.mean(FR_history[-t_off:],axis=0))
                    day_activity_ctx.append(np.mean(FR_history_ctx[-t_off:],axis=0))    
                    total_time += t_off
                    t_series.append(total_time)
                    nn.TurnOFF_FB()
                    nn.TurnOFF_FF()
                    for t in range(IR):
                        next_FR,FR_ctx, FR_op = nn.step(zero_input,0,op_inp)
                        FR_history.append(next_FR)
                        FR_history_ctx.append(FR_ctx)
                        FR_op_history.append(FR_op)
                        EX_history.append(nn.excitability.copy())
                        EX_history_ctx.append(nn.excitability_ctx.copy())
                        input_history.append(zero_input.copy())
                    total_time += IR
                    if rep != Nrep - 1:
                        t_series.append(total_time)
                rep_Activity.append(day_activity)
                rec_weights.append(nn.rec_w.copy())
                rec_ctx_weights.append(nn.rec_w_ctx.copy())
                mtl_op_weights.append(nn.mtl_op_w.copy())
                ctx_op_weights.append(nn.ctx_op_w.copy())
                # ff_weights.append(nn.input_w.copy())
                last_activity.append(np.mean(day_activity,axis=0))
                last_activity_ctx.append(np.mean(day_activity_ctx,axis=0))
                for t in range(ID):
                    next_FR,FR_ctx,FR_op = nn.step(zero_input,0,0)
                    FR_history.append(next_FR)
                    FR_history_ctx.append(FR_ctx)
                    FR_op_history.append(FR_op)
                    EX_history.append(nn.excitability.copy())
                    EX_history_ctx.append(nn.excitability_ctx.copy())
                    input_history.append(zero_input.copy())
                total_time += ID
                t_series.append(total_time)
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
        op_plot_folder = "./plots/{}".format(sim_name)
    
        os.makedirs(op_data_folder, exist_ok=True)
        np.save("{}/rec_weights.npy".format(op_data_folder),rec_weights_all)
        np.save("{}/mtl_op_weights.npy".format(op_data_folder),mtl_op_weights_all)
        np.save("{}/ctx_op_weights.npy".format(op_data_folder),ctx_op_weights_all)
        np.save("{}/rec_ctx_weights.npy".format(op_data_folder),rec_ctx_weights_all)

        np.save("{}/FR_history.npy".format(op_data_folder),FR_history_all)
        np.save("{}/FR_history_ctx.npy".format(op_data_folder),FR_ctx_history_all)
        np.save("{}/FR_history_op.npy".format(op_data_folder),FR_op_history_all)
        np.save("{}/last_activity.npy".format(op_data_folder),last_activity_all)
        np.save("{}/last_activity_ctx.npy".format(op_data_folder),last_activity_ctx_all)

        np.save("{}/EX_history.npy".format(op_data_folder),EX_history_all)
        np.save("{}/EX_history_ctx.npy".format(op_data_folder),EX_history_ctx_all)
        np.save("{}/input_history.npy".format(op_data_folder),input_history_all)
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
            "input_ramp_time": input_ramp_time,
            "input_ramp_width": input_ramp_width,
            "IR": IR,
            "Nrep": Nrep,
            "start_seed": start_seed,  # if you want reproducibility
            "max_e":max_e,
            "total_time": total_time,
            "dt": dt,
            "NUM_SIM": NUM_SIM,
            "off_days": off_days,
            "t_series": t_series,
            "E_mod": 0.0,
            "tau_IE": 0.01,
            "neurons_per_day": n_neurons_per_day
        }
        data = {
                "model_params": {
                    "dop":dob,
                    "n_MTL": nn.n_MTL,
                    "tau": nn.tau,
                    "dt": nn.dt,
                    "lr_hpc": nn.lr_hpc,
                    "decay_r_hpc": nn.decay_r_hpc,
                    "lr_ctx": nn.lr_ctx,
                    "decay_r_ctx": nn.decay_r_ctx,
                    "lr_op_MTL":nn.lr_op_MTL,
                    "lr_op_CTX":nn.lr_op_CTX,
                    "I0": nn.I0,
                    "I1": nn.I1,
                    "I2": nn.I2,
                    "mu_ex":mu_ex,
                    "sigma_ex":sigma_ex,
                    "ff":nn.FF_MTL_CTX,
                    "fb":nn.FB_CTX_MTL
                },
                "simulation_params": sim_params
            }

        filename = "{}/all_params.json".format(op_data_folder)

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"All parameters saved to {filename}")
        # last_activity_all=  (last_activity_all > threshold).astype(float)*last_activity_all
        # last_activity_ctx_all =  (last_activity_ctx_all > threshold).astype(float)*last_activity_ctx_all
        # plot_corr_matrix(last_activity_all[0], fname="{}/corr_matrix.svg".format(op_plot_folder))
        # plot_corr_matrix(last_activity_ctx_all[0], fname="{}/corr_matrix_ctx.svg".format(op_plot_folder))
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
            marker = "^",
            no_plot=True
        )
        # breakpoint()
        mean_corr_cxt, std_corr_cxt, per_sim_corr_cxt, idx_cxt = plot_mean_std_corr_over_time(
            last_activity_ctx_all ,                # shape: (sims, time, neurons)
            ref_time_idx=0,         # Encoding
            xlabels=xlabs,         # must match number of non-ref times
            include_ref_bar=True,
            title="Cell population \n activity correlation",
            fname="{}/encoding_vs_others_mean_std_ctx.svg".format(op_plot_folder),
            cmap = "Greens",
            no_plot=True
        )
        plot_population_vector_correlations(
            [mean_corr, mean_corr_cxt],
            [std_corr, std_corr_cxt],
            xlabels=[i for i in range(N_off_days)],
            labels=["HPC", "ACC"],
            fname="{}/encoding_vs_others_mean_std_HPC_CTX".format(op_plot_folder),
            title="Cell population \n activity correlation",
            colors=["tab:orange", "tab:blue"],
            markers=["^", "o"],
        )


        FR_history_th = (FR_history_all > threshold).astype(float)*FR_history_all
        FR_history_th_ctx = (FR_ctx_history_all > threshold).astype(float)*FR_ctx_history_all
        mean_DR.append([np.sum(1-mean_corr)/(N_off_days)])
        mean_DR_ctx.append([np.sum(1-mean_corr_cxt)/(N_off_days)])
        print("excitability boosts:", x1,x2)
    
        input_data_folder = "./data/{}".format(sim_name)
        op_plot_folder = "./plots/{}".format(sim_name)
        PlotAll(input_data_folder=input_data_folder, op_plot_folder=op_plot_folder)
    
    # plt.plot(E_fl,mean_DR,label="HPC")
    # plt.plot(E_fl_ctx,mean_DR_ctx,label="CTX")
    # plt.xlabel("Excitability boost amplitude")
    # plt.ylabel("Normalized drift rate")
    # plt.legend()
    # plt.savefig("{}/drift_rate_vs_excitability_boost.svg".format("."))
    breakpoint()


if __name__ == "__main__":
    main()
