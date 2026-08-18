import matplotlib.pyplot as plt
import json
import os
import numpy as np
from plotting_widget import *
from Utilities import average_firing_rates_with_active,ensamble_overlap
from tqdm import trange
start_seed = 120
np.random.seed(start_seed)


def softplus(x):
    """Numerically stable NumPy implementation of softplus."""
    return np.log1p(np.exp(np.clip(x, -500, 500)))


def relu(x):
    return np.maximum(0, x)


def tanh_window(t, duration, ramp_width=6.0, ramp_time=10.0):
    """Smoothly ramp input up and down within a stimulus presentation."""
    ramp_up = 0.5 * (1.0 + np.tanh((t - ramp_time) / ramp_width))
    ramp_down = 0.5 * (
        1.0 + np.tanh((duration - 1 - t - ramp_time) / ramp_width)
    )
    return ramp_up * ramp_down


def pca_transform_activity(activity_3d, n_components=3):
    """
    Fit PCA to day-averaged reactivation activity and return PC scores.

    Parameters
    ----------
    activity_3d : np.ndarray
        Shape (sims, days, neurons). For this script, this is last_activity_all.
    n_components : int
        Number of principal components to keep.
    """
    sims, days, neurons = activity_3d.shape
    data_2d = activity_3d.reshape(sims * days, neurons)
    mean_activity = data_2d.mean(axis=0, keepdims=True)
    centered = data_2d - mean_activity

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(n_components, vt.shape[0])
    components = vt[:n_components]
    scores_2d = centered @ components.T
    scores_3d = scores_2d.reshape(sims, days, n_components)

    denom = np.sum(singular_values**2)
    explained_variance_ratio = (
        (singular_values[:n_components] ** 2) / denom
        if denom > 0
        else np.zeros(n_components)
    )
    return scores_3d, components, mean_activity.squeeze(), explained_variance_ratio


def cosine_similarity_to_day0(scores_3d, eps=1e-12):
    """Cosine similarity between each day's PCA scores and day 0 per simulation."""
    ref = scores_3d[:, [0], :]
    numerator = np.sum(ref * scores_3d, axis=2)
    denominator = np.linalg.norm(ref, axis=2) * np.linalg.norm(scores_3d, axis=2)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > eps,
    )


class twolayer_FF:
    def __init__(self, n_inp, n_neurons,n_cont,baseline_e,tau=20.0, dt=1.0, act=None, lr=1/800, decay_r=1/1000, I0=1, I1=0.05, I2=0.001):
        if act is None:
            act = softplus
        self.n_neurons = n_neurons
        self.tau = tau  # rate constant 
        self.dt = dt  # discretization step
        self.act = act  # activation function
        self.lr = lr  # learning rate for synaptic weights
        self.decay_r = decay_r  # decay rate for synaptic weights
        self.I0 = I0
        self.I1 = I1
        self.I2 = I2
        # Keep an independent array so later excitability modulation cannot
        # mutate the baseline array supplied by the simulation.
        self.excitability = baseline_e.copy()
        self.plas_threshold = 0
        self.act_threshold = np.abs(np.random.normal(0,0.7,size=(n_neurons,)))
        self.n_cont = n_cont
        self.cont_exc = np.abs(np.random.normal(0,1,size=(self.n_cont,)))
        self.act_threshold_cont = np.abs(np.random.normal(0,0.7,size=(self.n_cont,)))
        # self.threshold = threshold
        
        # Initialize random input weights
        self.input_w = np.abs(np.random.normal(0,0.05,size=(n_inp,n_neurons)))
        # self.input_w = np.clip(self.input_w, 0.0, 0.1)
        self.rec_w = np.zeros((n_neurons, n_neurons))
        self.rec_w_cont = np.zeros((self.n_cont, self.n_cont))
        
        # Zero initial rate state
        self.rates_cont = np.zeros(self.n_cont)
        self.rates = np.zeros(n_neurons)
        # self.
    def step(self, input_FR,cont_INP):
        """
        Perform one timestep of rate dynamics:
        input_vector: shape [n_neurons]
        """
        # calculating the input to the RNN
        input_vector = input_FR + self.rec_w @ self.rates
        cont_inp = cont_INP + self.rec_w_cont @ self.rates_cont
        # breakpoint()
        # print(input_vector.max())
        # blanket inhibition to the RNN
        I_inhib = self.I0 + self.I1 * np.sum(self.rates) + self.I2 * np.sum(self.rates**2)
        I_inhib_cont = self.I0 + self.I1 * np.sum(self.rates_cont) + self.I2 * np.sum(self.rates_cont**2)


        # print(I_inhib)
        # total input to the RNN
        input_current =  input_vector -  I_inhib 
        input_cont = cont_inp - I_inhib_cont

        # rate change as the nonlinear ODE
        dr_dt = (-self.rates +   self.act(self.excitability + input_current + self.act_threshold)) / self.tau
        dr_dt_cont = (-self.rates_cont +   self.act(self.cont_exc + input_cont + self.act_threshold_cont)) / self.tau

        
        self.rates += (dr_dt * self.dt)
        self.rates_cont += (dr_dt_cont * self.dt)

        # print(self.rates.max())
        # post_mask = (self.rates > self.threshold).float()
        post_mask = self.rates > self.plas_threshold
        # hebbian plasticity in RNN weights
        hebbian_dw = self.lr * np.outer(self.rates*post_mask, self.rates) * self.dt
        decay = self.decay_r * self.rec_w * self.dt
        # hebbian plasticity in input weights
        self.rec_w += (hebbian_dw - decay)

        hebbian_dw_cont = self.lr * np.outer(self.rates_cont, self.rates_cont) * self.dt
        decay_cont = self.decay_r * self.rec_w_cont * self.dt

        self.rec_w_cont += (hebbian_dw_cont - decay_cont)

        # hebbian plasticity in RNN weights
        # hebbian_dw_inp = self.lr * np.outer(input_FR,self.rates*post_mask ) * self.dt
        # decay_inp = self.decay_r*0.1 * self.input_w * self.dt
        # # hebbian plasticity in input weights
        # self.input_w += (hebbian_dw_inp - decay_inp)

        # self.rates = np.clip(self.rates, 0.0, 15)  # Ensure rates are non-negative
        self.rec_w = np.clip(self.rec_w, 0.0, 1.0)  # Ensure weights are non-negative
        self.rec_w_cont = np.clip(self.rec_w_cont, 0.0, 1.0)
        # self.input_w = np.clip(self.input_w, 0.0, 0.2)  # Ensure weights are non-negative
        # self._normalize_input_outgoing(target_sum=15)
        return np.concatenate((self.rates_cont.copy(),self.rates.copy()))
    
    def _normalize_input_outgoing(self, target_sum=None, eps=1e-12):
        """Normalize columns so that for each input feature, the outgoing weights sum to target_sum."""
        if target_sum is None:
            target_sum = self.target_out_sum
        # columns correspond to inputs if input_w is (n_neurons, n_inp)
        col_sums = self.rec_w.sum(axis=1, keepdims=True)
        # avoid division by zero: if a column is all zeros, leave it unchanged
        scale = np.where(col_sums > eps, target_sum / col_sums, np.ones_like(col_sums))
        self.rec_w = self.rec_w * scale  # broadcast over rows


def main():
    FR_history = []
    EX_history = []
    rec_weights = []
    ff_weights = []
    last_activity = []
    input_history = []
    #
    n = 140
    n_inp = 140
    n_cont = 0
    E_fl = 1.3
    E_fe = 1.5
    E_ref = 0.7
    threshold = 2
    off_set = 0

    # base_E[:off_set] += 2
    FC_inp = 25
    input = 18*np.ones(n_inp)
    cont_inp = 13*np.ones(n_cont)
    zero_cont_inp = np.zeros(n_cont)
    # input[:10] = FC_inp
    off_input = input#18*np.ones(n_inp)
    # recall_input = off_input.clone()
    # recall_input[:20] = FC_inp
    # off_input[:off_set] -= 0
    noisy_input = np.random.normal(0,1,size=(n_inp,))
    # input = noisy_input
    zero_input = np.zeros(n_inp)

    ID = 1000

    NUM_SIM = 15
    N_off_days = 7
    t_off = 100
    input_ramp_time = 10.0
    input_ramp_width = 6.0
    IR = 100
    Nrep = 10
    n_stimulus_neurons = 10
    stimulus_neuron_indices = np.arange(n_stimulus_neurons, dtype=int)
    stimulus_input_amplitude = 20.0
    other_input_amplitude = 18.0
    input[:] = other_input_amplitude
    input[stimulus_neuron_indices] = stimulus_input_amplitude
    offline_candidate_indices = np.arange(n_stimulus_neurons, n, dtype=int)
    offline_reactivation_fraction = 0.50
    n_offline_reactivated = int(
        round(offline_candidate_indices.size * offline_reactivation_fraction)
    )
    data_folder = "./data/Reimagined_3_random50"
    plot_folder = "./plots/Reimagined_3_random50"

    FR_history_all = []
    EX_history_all = []
    rec_weights_all = []
    ff_weights_all = []
    last_activity_all = []
    input_history_all = []
    reactivation_masks_all = []

    for i in trange(NUM_SIM):
        np.random.seed(start_seed + i)
        base_E = np.abs(np.random.normal(0,1,size=(n,)))
        nn = twolayer_FF(n_inp=n_inp,
                        n_neurons=n,
                        n_cont=n_cont, 
                        baseline_e = base_E, 
                        tau=20.0, dt=1, 
                        act=relu, 
                        lr=1/800, decay_r=1/1000,
                        I0=8, I1=0.7, I2=0.05)
        FR_history = []
        EX_history = []
        rec_weights = []
        ff_weights = []
        last_activity = []
        input_history = []
        reactivation_masks = []
        rep_Activity = []
        high_threshold = 5
        for day in range(N_off_days):
            day_activity = []
            # Reset to the original baseline before boosting the current day's
            # cohort. The copy prevents boosts from accumulating in base_E.
            nn.excitability = base_E.copy()
            nn.excitability[off_set+(day)*20:off_set+(day)*20+20] += E_fl
            # Encoding (day 0) and recall (final day) use the full cue. Each
            # intervening offline session independently reactivates a random
            # 50% of the non-stimulus population without replacement. The
            # reserved stimulus neurons are never externally driven during
            # offline reactivation.
            if day == 0 or day == N_off_days - 1:
                reactivation_mask = np.ones(n, dtype=bool)
            else:
                reactivation_mask = np.zeros(n, dtype=bool)
                reactivated = np.random.choice(
                    offline_candidate_indices,
                    size=n_offline_reactivated,
                    replace=False,
                )
                reactivation_mask[reactivated] = True
                if np.any(reactivation_mask[stimulus_neuron_indices]):
                    raise RuntimeError(
                        "Stimulus neurons were selected for offline reactivation"
                    )
            reactivation_masks.append(reactivation_mask.copy())
            inp_to_network = input * reactivation_mask
            if day == 0 or day == 6:
                cont_inp_to_network = cont_inp
            else:
                cont_inp_to_network = zero_cont_inp
                
            for rep in range(Nrep):
                for t in range(t_off):
                    ramp = tanh_window(
                        t,
                        t_off,
                        ramp_width=input_ramp_width,
                        ramp_time=input_ramp_time,
                    )
                    ramped_input = ramp * inp_to_network
                    ramped_cont_input = ramp * cont_inp_to_network
                    next_FR = nn.step(ramped_input, ramped_cont_input)
                    FR_history.append(next_FR)
                    EX_history.append(nn.excitability.copy())
                    input_history.append(ramped_input.copy())
                day_activity.append(np.mean(FR_history[-t_off:],axis=0))
                for t in range(IR):
                    next_FR = nn.step(zero_input,zero_cont_inp)
                    FR_history.append(next_FR)
                    EX_history.append(nn.excitability.copy())
                    input_history.append(zero_input)
            rep_Activity.append(day_activity)
            rec_weights.append(nn.rec_w.copy())
            # ff_weights.append(nn.input_w)
            last_activity.append(np.mean(day_activity,axis=0))
            for t in range(ID):
                next_FR = nn.step(zero_input,zero_cont_inp)
                FR_history.append(next_FR)
                EX_history.append(nn.excitability.copy())
                input_history.append(zero_input)
        # breakpoint()
        FR_history_all.append(FR_history)
        EX_history_all.append(EX_history)
        rec_weights_all.append(rec_weights)
        ff_weights_all.append(ff_weights)
        last_activity_all.append(last_activity)
        input_history_all.append(input_history)
        reactivation_masks_all.append(reactivation_masks)

    FR_history_all = np.stack(FR_history_all)
    # input_history = np.stack(input_history)
    EX_history_all = np.stack(EX_history_all)
    last_activity_all = np.stack(last_activity_all)
    input_history_all = np.stack(input_history_all)
    reactivation_masks_all = np.stack(reactivation_masks_all)
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)
    np.save(os.path.join(data_folder, "FR_history.npy"), FR_history_all)
    np.save(os.path.join(data_folder, "EX_history.npy"), EX_history_all)
    np.save(os.path.join(data_folder, "last_activity.npy"), last_activity_all)
    np.save(os.path.join(data_folder, "input_history.npy"), input_history_all)
    np.save(
        os.path.join(data_folder, "offline_reactivation_masks.npy"),
        reactivation_masks_all,
    )

    pca_n_components = 3
    pca_scores, pca_components, pca_mean_activity, pca_explained_variance = pca_transform_activity(
        last_activity_all,
        n_components=pca_n_components,
    )
    pca_cosine_to_day0 = cosine_similarity_to_day0(pca_scores)
    np.save(os.path.join(data_folder, "pca_scores_reactivation.npy"), pca_scores)
    np.save(os.path.join(data_folder, "pca_components_reactivation.npy"), pca_components)
    np.save(os.path.join(data_folder, "pca_mean_activity_reactivation.npy"), pca_mean_activity)
    np.save(os.path.join(data_folder, "pca_explained_variance_reactivation.npy"), pca_explained_variance)
    np.save(os.path.join(data_folder, "pca_cosine_similarity_to_day0.npy"), pca_cosine_to_day0)

    cos_mean = np.nanmean(pca_cosine_to_day0, axis=0)
    cos_sem = np.nanstd(pca_cosine_to_day0, axis=0) / np.sqrt(NUM_SIM)
    cos_x = np.arange(N_off_days)
    cos_labels = ["Day 0"] + [f"Off {i}" for i in range(1, N_off_days - 1)] + ["Recall"]
    plt.figure(figsize=(7, 4))
    plt.errorbar(cos_x, cos_mean, yerr=cos_sem, marker="o", capsize=4)
    plt.axhline(0, color="k", linewidth=0.8, alpha=0.4)
    plt.xticks(cos_x, cos_labels, rotation=45, ha="right")
    plt.ylabel("Cosine similarity to day 0")
    plt.xlabel("Day")
    plt.title("PCA reactivation similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, "pca_cosine_similarity_to_day0"), dpi=300)
    plt.close()

    sim_params = {
        "n": n,
        "n_inp": n_inp,
        "E_fl": E_fl,
        "threshold": threshold,
        "ID": ID,
        "N_off_days": N_off_days,
        "t_off": t_off,
        "input_ramp_time": input_ramp_time,
        "input_ramp_width": input_ramp_width,
        "IR": IR,
        "Nrep": Nrep,
        "offline_reactivation_fraction": offline_reactivation_fraction,
        "n_offline_reactivated": n_offline_reactivated,
        "n_stimulus_neurons": n_stimulus_neurons,
        "stimulus_neuron_indices": stimulus_neuron_indices.tolist(),
        "stimulus_input_amplitude": stimulus_input_amplitude,
        "other_input_amplitude": other_input_amplitude,
        "pca_n_components": pca_n_components,
        "start_seed": start_seed  # if you want reproducibility
    }
    data = {
            "model_params": {
                "n_neurons": nn.n_neurons,
                "tau": nn.tau,
                "dt": nn.dt,
                "lr": nn.lr,
                "decay_r": nn.decay_r,
                "I0": nn.I0,
                "I1": nn.I1,
                "I2": nn.I2,
                # "excitability": model.excitability.detach().cpu().numpy().tolist(),
            },
            "simulation_params": sim_params
        }
    filename = os.path.join(data_folder, "all_params.json")

    with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        # print(f"All parameters saved to {filename}")
    plot_corr_matrix(last_activity, fname=os.path.join(plot_folder, "corr_matrix"))

    # FR_history_th = (FR_history_all > threshold).astype(float)*FR_history_all
    # plot_activity_n_excitability_time([FR_history_th[0].T,EX_history_all[0].T],
    #                        titles=['Neuronal Activity',
    #                                 "Neuronal Excitability"],
    #                        fname="./plots/Reimagined_3/Activity_n_excitability",
    #                        cmaps=['Blues', 'Greens'])
    # labs = ["FC"] + [f"Off {i+1}" for i in range(N_off_days)]
    # plot_weights_over_time(rec_weights_all[0],
    #                        titles=  labs,
    #                        fname="./plots/Reimagined_3/Rec_w",
    #                        cmaps='gray_r')

    # Engram persistence analysis across the complete returned population. The
    # first n_cont entries are control/contextual neurons and the remaining
    # entries are the main recurrent population; both are included here.
    engram_activity = last_activity_all
    active_by_session = engram_activity > threshold
    first_session_rates = engram_activity[:, 0, :]
    first_session_engram = active_by_session[:, 0, :]
    active_session_counts = np.sum(active_by_session, axis=1)

    # Each retained sample is one (simulation, neuron) pair that belonged to the
    # session-0 engram. Counts include session 0 and therefore range from 1 to 7.
    first_engram_rates = first_session_rates[first_session_engram]
    first_engram_session_counts = active_session_counts[first_session_engram]
    np.save(
        os.path.join(data_folder, "first_session_engram_firing_rates.npy"),
        first_engram_rates,
    )
    np.save(
        os.path.join(data_folder, "first_session_engram_active_session_counts.npy"),
        first_engram_session_counts,
    )

    scatter_rates, scatter_counts, persistence_r = (
        plot_first_activity_vs_active_sessions(
            engram_activity,
            threshold=threshold,
            first_session_idx=0,
            mode="concat",
            include_ref_in_count=True,
            fname=os.path.join(plot_folder, "first_session_rate_vs_engram_sessions"),
        )
    )
    # Explicitly show every possible persistence bin. Empty bins remain visible
    # on the x-axis but have no bar. Statistics pool the session-0 engram neurons
    # from all simulations within each active-session-count bin.
    count_bins = np.arange(1, N_off_days + 1)
    mean_rates = np.full(count_bins.size, np.nan)
    sem_rates = np.full(count_bins.size, np.nan)
    samples_per_bin = np.zeros(count_bins.size, dtype=int)
    for bin_index, session_count in enumerate(count_bins):
        rates_in_bin = first_engram_rates[
            first_engram_session_counts == session_count
        ]
        samples_per_bin[bin_index] = rates_in_bin.size
        if rates_in_bin.size:
            mean_rates[bin_index] = np.mean(rates_in_bin)
            sem_rates[bin_index] = (
                np.std(rates_in_bin, ddof=1) / np.sqrt(rates_in_bin.size)
                if rates_in_bin.size > 1 else 0.0
            )

    fig, ax = plt.subplots(figsize=(8, 4))
    occupied = samples_per_bin > 0
    ax.bar(
        count_bins[occupied],
        mean_rates[occupied],
        yerr=sem_rates[occupied],
        capsize=5,
        color="skyblue",
        edgecolor="black",
    )
    for x_pos, y_pos, sample_count in zip(
        count_bins[occupied], mean_rates[occupied], samples_per_bin[occupied]
    ):
        ax.text(x_pos, y_pos, "n={}".format(sample_count), ha="center", va="bottom")
    ax.set_xticks(count_bins)
    ax.set_xlabel("Number of active sessions")
    ax.set_ylabel("Mean firing rate during session 1")
    ax.set_title("Session-1 firing rate by engram persistence")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_plot(os.path.join(plot_folder, "first_session_rate_by_engram_sessions"))
    plt.close(fig)

    persistence_summary = {
        "threshold": float(threshold),
        "number_of_first_session_engram_samples": int(first_engram_rates.size),
        "pearson_r": (
            None if np.isnan(persistence_r) else float(persistence_r)
        ),
        "active_session_count_bins": count_bins.tolist(),
        "mean_first_session_firing_rate": [
            None if np.isnan(value) else float(value) for value in mean_rates
        ],
        "sem_first_session_firing_rate": [
            None if np.isnan(value) else float(value) for value in sem_rates
        ],
        "neurons_per_bin": samples_per_bin.tolist(),
    }
    with open(
        os.path.join(data_folder, "engram_persistence_summary.json"), "w"
    ) as handle:
        json.dump(persistence_summary, handle, indent=4)

    print(
        "First-session engram samples:", first_engram_rates.size,
        "Pearson r:", persistence_r,
    )

    cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
    xlabs = ["Off 1","Off 2","Off 3","Off 4","Off 5","Recall"]
    Title = "Ensemble similarity of encoding and offline + recall"
    # plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined_3/encoding_corr", use_bar_plot=True)
    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_all,                # shape: (sims, time, neurons)
        ref_time_idx=0,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=False,
        title="Encoding vs. others (mean ± SD across sims)",
        fname=os.path.join(plot_folder, "encoding_vs_others_mean_std"),
        cmap="Reds",
        bar_plot=True
    )

    xlabs = ["Encoding", "Off 1","Off 2","Off 3","Off 4","Off 5"]
    Title = "Ensemble similarity of recall and offline + encoding"
    # plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined_3//Recall_corr", use_bar_plot=True)
    last_activity_all, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_all,                # shape: (sims, time, neurons)
        ref_time_idx=-1,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=False,
        title="Encoding vs. others (mean ± SD across sims)",
        fname=os.path.join(plot_folder, "recall_vs_others_mean_std"),
        cmap="Reds",
        bar_plot=True

    )


if __name__ == "__main__":
    main()
