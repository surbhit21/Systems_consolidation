import numpy as np
import os
from plotting_widget import (
    PLOT_LABEL_FONTSIZE,
    PLOT_TICK_FONTSIZE,
    PLOT_TITLE_FONTSIZE,
    plot_activity_n_excitability_time,
    plot_first_activity_vs_active_sessions,
    plot_mean_std_corr_over_time,
    plot_sessions_count_vs_activity_sem,
    plot_weights_over_time,
    save_plot,
    style_axis,
)
import matplotlib.pyplot as plt

threshold = 2
n_contextual_neurons = 6
# Change these two variables to choose the input dataset and plot destination.
input_data_folder = "./data/Reimagined_3"
plot_output_folder = "./plots/Reimagined_3"
os.makedirs(plot_output_folder, exist_ok=True)

# Timing of one simulated session in GetDrift_FF_random50.py.
Nrep = 10
t_off = 100
IR = 100
ID = 1000
time_per_day = Nrep * (t_off + IR) + ID
session_labels = [
    "Encoding",
    "Off 1",
    "Off 2",
    "Off 3",
    "Off 4",
    "Off 5",
    "Recall",
]

last_activity_all = np.load("{}/last_activity.npy".format(input_data_folder)) # shape: (sims, time, neurons)
if last_activity_all.shape[2] <= n_contextual_neurons:
    raise ValueError(
        "last_activity_all does not contain units beyond the configured "
        "contextual population"
    )
mnemonic_activity_all = last_activity_all[:, :, n_contextual_neurons:]
FR_history_all = np.load("{}/FR_history.npy".format(input_data_folder)) # shape: (sims, time, neurons)
EX_history_all = np.load("{}/EX_history.npy".format(input_data_folder)) # shape:
FR_history_th = (FR_history_all > threshold).astype(float)*FR_history_all
# breakpoint()
plot_activity_n_excitability_time([FR_history_th[0].T,EX_history_all[0].T],
                       titles=['Neuronal Activity',
                                "Neuronal Excitability"],
                       fname=os.path.join(plot_output_folder, "Activity_n_excitability"),
                       cmaps=['Blues', 'Greens'],
                       colorbar_label=['Firing rate (Hz)', 'Excitability (a.u.)'],
                       time_per_day=time_per_day,
                       day_zero_time=0,
                       day_labels=session_labels,
                       ncols=1,
                       figsize=(16, 7))

# Recurrent matrices are saved by GetDrift_FF.py at the end of each session.
recurrent_weights_path = os.path.join(input_data_folder, "rec_weights.npy")
if os.path.exists(recurrent_weights_path):
    recurrent_weights_all = np.load(recurrent_weights_path)
    if recurrent_weights_all.ndim != 4:
        raise ValueError(
            "rec_weights.npy must have shape "
            "(simulations, sessions, neurons, neurons)"
        )
    if recurrent_weights_all.shape[1] != len(session_labels):
        raise ValueError(
            "The number of saved recurrent-weight sessions does not match "
            "session_labels"
        )
    plot_weights_over_time(
        recurrent_weights_all[0],
        titles=session_labels,
        title_fontsize=26,
        tick_fontsize=22,
        colorbar_fontsize=22,
        fname=os.path.join(plot_output_folder, "recurrent_weights"),
        cmaps="gray_r",
        plot_title="Recurrent weights across sessions",
        colorbar_label="Recurrent weight (a.u.)",
    )
else:
    print(
        f"Skipping recurrent-weight plot: {recurrent_weights_path} was not "
        "found. Rerun GetDrift_FF.py to generate it."
    )

cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
xlabs = ["Off 1","Off 2","Off 3","Off 4","Off 5","Recall"]
Title = "Ensemble similarity"
# plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr", use_bar_plot=True)
mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    last_activity_all ,                # shape: (sims, time, neurons)
    ref_time_idx=0,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=False,
    title="Ensemble similarity with Encoding",
    fname=os.path.join(plot_output_folder, "encoding_vs_others_mean_std"),
    cmap="Reds",
    bar_plot=True,
    xlab = "Sessions"
)

# Mnemonic population only: exclude the leading contextual units before
# calculating correlations across encoding, offline sessions, and recall.
mnemonic_encoding_corr, mnemonic_encoding_sem, _, _ = (
    plot_mean_std_corr_over_time(
        mnemonic_activity_all,
        ref_time_idx=0,
        xlabels=["Off 1", "Off 2", "Off 3", "Off 4", "Off 5", "Recall"],
        include_ref_bar=False,
        title="Mnemonic ensemble similarity with Encoding",
        fname=os.path.join(
            plot_output_folder,
            "mnemonic_encoding_vs_offline_and_recall",
        ),
        cmap="Reds",
        bar_plot=True,
        xlab="Sessions",
    )
)

mnemonic_recall_corr, mnemonic_recall_sem, _, _ = (
    plot_mean_std_corr_over_time(
        mnemonic_activity_all,
        ref_time_idx=-1,
        xlabels=["Encoding", "Off 1", "Off 2", "Off 3", "Off 4", "Off 5"],
        include_ref_bar=False,
        title="Mnemonic ensemble similarity with Recall",
        fname=os.path.join(
            plot_output_folder,
            "mnemonic_recall_vs_encoding_and_offline",
        ),
        cmap="Reds",
        bar_plot=True,
        xlab="Sessions",
    )
)

xlabs = ["Encoding", "Off 1","Off 2","Off 3","Off 4","Off 5"]
Title = "Ensemble similarity"
# plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined//Recall_corr", use_bar_plot=True)
mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    last_activity_all,                # shape: (sims, time, neurons)
    ref_time_idx=-1,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=False,
    title="Ensemble similarity with Recall",
    fname=os.path.join(plot_output_folder, "recall_vs_others_mean_std"),
    cmap="Reds",
    bar_plot=True,
    xlab = "Sessions"

)


cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
xlabs = ["Off 1","Off 2","Off 3","Off 4","Off 5"]
Title = "Ensemble similarity"
# breakpoint()
# plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr", use_bar_plot=True)
mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    last_activity_all[:,:-1,:] ,                # shape: (sims, time, neurons)
    ref_time_idx=0,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=False,
    title="Ensemble similarity with Encoding",
    fname=os.path.join(plot_output_folder, "encoding_vs_offline_mean_std"),
    cmap="Reds",
    bar_plot=True,
    xlab = "Sessions"
)

xlabs = ["Off 1","Off 2","Off 3","Off 4","Off 5"]
Title = "Ensemble similarity"
# plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined//Recall_corr", use_bar_plot=True)
mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    last_activity_all[:,1:,:],                # shape: (sims, time, neurons)
    ref_time_idx=-1,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=False,
    title="Ensemble similarity with Recall",
    fname=os.path.join(plot_output_folder, "recall_vs_offline_mean_std"),
    cmap="Reds",
    bar_plot=True,
    xlab = "Sessions"

)

S, T, N = last_activity_all.shape

# Treat NaNs as "not active" (change if you prefer to ignore them)
active = np.where(np.isnan(last_activity_all), False, last_activity_all > threshold)  # (S, T, N)

first_active = active[:, 0, :]          # (S, N) first session
other_active = active[:, 1:, :]         # (S, T-1, N) all sessions after the first

# Intersection & union with the first session, per sim & session
intersection = np.logical_and(first_active[:, None, :], other_active).sum(axis=-1)   # (S, T-1)
union        = np.logical_or (first_active[:, None, :], other_active).sum(axis=-1)   # (S, T-1)

# 1) Raw overlap counts
overlap_counts = intersection                                                # (S, T-1)

# 2) Fraction of first-session actives recovered later (recall of first set)
first_counts = first_active.sum(axis=-1)[:, None]                            # (S, 1)
overlap_frac_first = np.divide(
    intersection, first_counts,
    out=np.zeros_like(intersection, dtype=float), where=first_counts > 0
)                                                                           # (S, T-1)

mean_frac_first = overlap_frac_first.mean(axis=0)             # (T-1,)
std_frac_first  = overlap_frac_first.std(axis=0, ddof=0)  
sem_frac_first  = std_frac_first/np.sqrt(S)           # (T-1,)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(T - 1)
overlap_labels = ["Off 1", "Off 2", "Off 3", "Off 4", "Off 5", "Recall"]
overlap_cmap = plt.get_cmap("Reds")
overlap_colors = [overlap_cmap(i / T) for i in range(T - 1)]
ax.bar(
    x,
    mean_frac_first,
    yerr=sem_frac_first,
    color=overlap_colors,
)
ax.spines[["right", "top"]].set_visible(False)
style_axis(
    ax,
    title="Encoding ensemble reactivation across offline sessions",
    xlabel="Sessions",
    ylabel="Overlap fraction",
    title_fontsize=PLOT_TITLE_FONTSIZE,
    label_fontsize=PLOT_LABEL_FONTSIZE,
    tick_fontsize=PLOT_TICK_FONTSIZE,
)
ax.set_xticks(x, labels=overlap_labels)
ax.set_ylim(0, 1.1)
ax.tick_params(
    axis="both", which="major", labelsize=PLOT_TICK_FONTSIZE
)
fig.tight_layout()
save_plot(os.path.join(plot_output_folder, "overlap_frac_first_mean_sem"))
plt.close(fig)

# Companion figure restricted to the five offline sessions (recall excluded).
offline_mean = mean_frac_first[:-1]
offline_sem = sem_frac_first[:-1]
offline_x = np.arange(len(offline_mean))
offline_labels = overlap_labels[:-1]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(
    offline_x,
    offline_mean,
    yerr=offline_sem,
    color=[overlap_cmap(i / T) for i in range(len(offline_mean))],
)
ax.spines[["right", "top"]].set_visible(False)
style_axis(
    ax,
    title="Encoding ensemble reactivation across offline sessions",
    xlabel="Sessions",
    ylabel="Overlap fraction",
    title_fontsize=PLOT_TITLE_FONTSIZE,
    label_fontsize=PLOT_LABEL_FONTSIZE,
    tick_fontsize=PLOT_TICK_FONTSIZE,
)
ax.set_xticks(offline_x, labels=offline_labels)
ax.set_ylim(0, 0.7)
ax.tick_params(
    axis="both", which="major", labelsize=PLOT_TICK_FONTSIZE
)
fig.tight_layout()
save_plot(
    os.path.join(plot_output_folder, "overlap_frac_first_mean_sem_no_recall")
)
plt.close(fig)


x1, y1, r1 = plot_first_activity_vs_active_sessions(
    last_activity_all, threshold=threshold,
    first_session_idx=0,
    mode="concat",
    include_ref_in_count=True,
    fname=os.path.join(plot_output_folder, "first_activity_vs_counts_concat"),
)
print(r1)

counts_x, mean_y, sem_y, n = plot_sessions_count_vs_activity_sem(
    last_activity_all, threshold=threshold,
    first_session_idx=0,
    include_ref_in_count=True,
    sem_mode='pooled',
    fname=os.path.join(plot_output_folder, "first_act_vs_sessions_pooled")
)
