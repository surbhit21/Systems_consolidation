import numpy as np
from plotting_widget import *
import matplotlib.pyplot as plt
import json
input_data_folder = "./data/CNT_slow_drift_with_IP_with_FB"
op_plots_folder = "./plots/CNT_slow_drift_with_IP_with_FB"
# Path to your JSON file
json_path = "{}/all_params.json".format(input_data_folder)
dop = 2
with open(json_path, "r") as f:
    sim_params = json.load(f)

# breakpoint()
sim_params = sim_params["simulation_params"]
# Explicitly assign variables
n = sim_params["n"]
n_inp = sim_params["n_inp"]
n_ctx = sim_params["n_ctx"]
E_fl = sim_params["E_fl"]
FC_inp = sim_params["FC_inp"]
E_fl_ctx = sim_params["E_fl_ctx"]
threshold = sim_params["threshold"]
ID = sim_params["ID"]
N_off_days = sim_params["N_off_days"]
t_off = sim_params["t_off"]
IR = sim_params["IR"]
Nrep = sim_params["Nrep"]
start_seed = sim_params["start_seed"]
max_e = sim_params["max_e"]
total_time = sim_params["total_time"]
dt = sim_params["dt"]
# import
# threshold = 2

last_activity_all = np.load("{}/last_activity.npy".format(input_data_folder)) # shape: (sims, time, neurons)
FR_history_all = np.load("{}/FR_history.npy".format(input_data_folder)) # shape: (sims, time, neurons)
EX_history_all = np.load("{}/EX_history.npy".format(input_data_folder)) # shape:
FR_op_history_all = np.load("{}/FR_history_op.npy".format(input_data_folder))
last_activity_all_ctx = np.load("{}/last_activity_ctx.npy".format(input_data_folder)) # shape: (sims, time, neurons)
FR_history_all_ctx = np.load("{}/FR_history_ctx.npy".format(input_data_folder)) # shape: (sims, time, neurons)
EX_history_all_ctx = np.load("{}/EX_history_ctx.npy".format(input_data_folder)) # shape: 
rec_weights_all = np.load("{}/rec_weights.npy".format(input_data_folder))
rec_ctx_weights_all = np.load("{}/rec_ctx_weights.npy".format(input_data_folder))
FR_history_th = (FR_history_all > threshold).astype(float)*FR_history_all
FR_history_th_ctx = (FR_history_all_ctx > threshold).astype(float)*FR_history_all_ctx
last_activity_th = (last_activity_all > threshold).astype(float)*last_activity_all
last_activity_th_ctx = (last_activity_all_ctx > threshold).astype(float)*last_activity_all_ctx
breakpoint()
# total_time = 22000
timepoints = np.arange(0,total_time,1)*1
plot_firing_rate(timepoints, FR_op_history_all[:, :, 0],lab = "Output neuron",
                 xlabel="Time (s)", ylabel="Firing Rate (Hz)", c="r",fname= "{}/OP_neuron_activity.svg".format(op_plots_folder))
# breakpoint()
plot_activity_n_excitability_time([FR_history_th[0].T,EX_history_all[0].T],
                       titles=['Neuronal Activity',
                                "Neuronal Excitability"],
                       fname="{}/Activity_n_excitability.svg".format(op_plots_folder),
                       cmaps=['Oranges', 'Blues'])
plot_activity_n_excitability_time([FR_history_th_ctx[0].T,EX_history_all_ctx[0].T],
                       titles=['Neuronal Activity',
                                "Neuronal Excitability"],
                       fname="{}/Activity_n_excitability_ctx.svg".format(op_plots_folder),
                       cmaps=['Greens', 'Blues'])
# labs = ["FC"] t [f"Off {it1}" for i in range(N_off_days)]
# plot_weights_over_time(rec_weights_all[0],
#                        titles=  labs,
#                        fname="./plots/Reimagined/Rec_w.svg",
#                        cmap='gray_r')

cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
xlabs = [f"{i}" for i in range(N_off_days)]
Title = "Ensemble similarity"
# plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr.svg", use_bar_plot=True)
mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    last_activity_th ,                # shape: (sims, time, neurons)
    ref_time_idx=0,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=True,
    title="Cell population \n activity correlation",
    fname="{}/encoding_vs_others_mean_std.svg".format(op_plots_folder),
    cmap = "Oranges",
    marker = "^"
)
mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    last_activity_th_ctx ,                # shape: (sims, time, neurons)
    ref_time_idx=0,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=True,
    title="Cell population \n activity correlation",
    fname="{}/encoding_vs_others_mean_std_ctx.svg".format(op_plots_folder),
    cmap = "Greens"
)

xlabs = [f"{i-dop}" for i in range(N_off_days)]
mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    last_activity_th ,                # shape: (sims, time, neurons)
    ref_time_idx=dop,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=True,
    title="Cell population \n activity correlation",
    fname="{}/off1_vs_others_mean_std.svg".format(op_plots_folder),
    cmap = "Oranges",
    marker = "^"
)

mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    last_activity_th_ctx ,                # shape: (sims, time, neurons)
    ref_time_idx=dop,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=True,
    title="Cell population \n activity correlation",
    fname="{}/off1_vs_others_mean_std_ctx.svg".format(op_plots_folder),
    cmap = "Greens"
)

xlabs = [f"{i}" for i in range(N_off_days)]
Title = "Ensemble similarity"
# plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined//Recall_corr.svg", use_bar_plot=True)
mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    last_activity_th,                # shape: (sims, time, neurons)
    ref_time_idx=-1,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=True,
    title="Cell population \n activity correlation",
    fname="{}/recall_vs_others_mean_std.svg".format(op_plots_folder),
    cmap = "Oranges",
    marker = "^"

)

mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    last_activity_th_ctx,                # shape: (sims, time, neurons)
    ref_time_idx=-1,         # Encoding
    xlabels=xlabs,         # must match number of non-ref times
    include_ref_bar=True,
    title="Cell population \n activity correlation",
    fname="{}/recall_vs_others_mean_std_ctx.svg".format(op_plots_folder),
    cmap = "Greens",
    

)

# cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
# xlabs = ["Off 1","Off 2","Off 3"]
# Title = "Ensemble similarity"
# # breakpoint()
# # plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr.svg", use_bar_plot=True)
# mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
#     last_activity_all[:,:-1,:] ,                # shape: (sims, time, neurons)
#     ref_time_idx=0,         # Encoding
#     xlabels=xlabs,         # must match number of non-ref times
#     include_ref_bar=False,
#     title="Cell population \n activity correlation",
#     fname="{}/encoding_vs_offline_mean_std.svg".format(op_plots_folder),
#     cmap = "Oranges",
#     marker = "^"
# )
# mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
#     last_activity_all_ctx[:,:-1,:] ,                # shape: (sims, time, neurons)
#     ref_time_idx=0,         # Encoding
#     xlabels=xlabs,         # must match number of non-ref times
#     include_ref_bar=False,
#     title="Cell population \n activity correlation",
#     fname="{}/encoding_vs_offline_mean_std_ctx.svg".format(op_plots_folder),
#     cmap = "Greens"
# )

# xlabs = [f"Off {i+1}" for i in range(N_off_days)]
# # Title = "Ensemble similarity"
# # # plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined//Recall_corr.svg", use_bar_plot=True)
# # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
# #     last_activity_all[:,1:,:],                # shape: (sims, time, neurons)
# #     ref_time_idx=-1,         # Encoding
# #     xlabels=xlabs,         # must match number of non-ref times
# #     include_ref_bar=False,
# #     title="Cell population \n activity correlation",
# #     fname="{}/recall_vs_offline_mean_std.svg".format(op_plots_folder),
# #     cmap = "Oranges",
# #     marker = "^"

# # )

# # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
# #     last_activity_all_ctx[:,1:,:],                # shape: (sims, time, neurons)
# #     ref_time_idx=-1,         # Encoding
# #     xlabels=xlabs,         # must match number of non-ref times
# #     include_ref_bar=False,
# #     title="Cell population \n activity correlation",
# #     fname="{}/recall_vs_offline_mean_std_ctx.svg".format(op_plots_folder),
# #     cmap = "Greens"

# # )

# S, T, N = last_activity_all.shape

# # Treat NaNs as "not active" (change if you prefer to ignore them)
# active = np.where(np.isnan(last_activity_all), False, last_activity_all > threshold)  # (S, T, N)

# first_active = active[:, 0, :]          # (S, N) first session
# other_active = active[:, 1:, :]         # (S, T-1, N) all sessions after the first

# # Intersection & union with the first session, per sim & session
# intersection = np.logical_and(first_active[:, None, :], other_active).sum(axis=-1)   # (S, T-1)
# union        = np.logical_or (first_active[:, None, :], other_active).sum(axis=-1)   # (S, T-1)

# # 1) Raw overlap counts
# overlap_counts = intersection                                                # (S, T-1)

# # 2) Fraction of first-session actives recovered later (recall of first set)
# first_counts = first_active.sum(axis=-1)[:, None]                            # (S, 1)
# overlap_frac_first = np.divide(
#     intersection, first_counts,
#     out=np.zeros_like(intersection, dtype=float), where=first_counts > 0
# )                                                                           # (S, T-1)

# mean_frac_first = overlap_frac_first.mean(axis=0)             # (T-1,)
# std_frac_first  = overlap_frac_first.std(axis=0, ddof=0)  
# sem_frac_first  = std_frac_first/np.sqrt(S)           # (T-1,)

# fig, ax = plt.subplots(figsize=(8, 4))
# x = np.arange(1,T - 1)
# font_size = 14
# tick_fontsize = 12
                                         
# ax.bar(x, mean_frac_first[:-1], yerr=sem_frac_first[:-1], capsize=5,  edgecolor='black', alpha=0.9)
# # Cosmetics
# ax.spines[["right", "top"]].set_visible(False)
# ax.set_title("Cell overlap fraction with encoding", fontsize=font_size)
# ax.set_xlabel("Session", fontsize=font_size)
# ax.set_ylabel("Overlap fraction", fontsize=font_size)
# ax.set_xticks(x, labels=xlabs)
# ax.tick_params(labelsize=tick_fontsize)
# ax.set_ylim(0, 0.5)
# # ax.grid(True, axis='y', linestyle='--', alpha=0.35)
# fig.tight_layout()
# plt.savefig("{}/overlap_frac_first_mean_sem.svg".format(op_plots_folder))
# plt.show()


# x1, y1, r1 = plot_first_activity_vs_active_sessions(
#     last_activity_all, threshold=threshold,
#     first_session_idx=0,
#     mode="concat",
#     include_ref_in_count=True,
#     fname="{}/first_activity_vs_counts_concat.svg".format(op_plots_folder)
# )
# print(r1)

# counts_x, mean_y, sem_y, n = plot_sessions_count_vs_activity_sem(
#     last_activity_all, threshold=threshold,
#     first_session_idx=0,
#     include_ref_in_count=True,
#     sem_mode='pooled',
#     fname="{}/first_act_vs_sessions_pooled.svg".format(op_plots_folder)
# )
# labs = ["Encoding","Day 1","Day 2","Day 3","Day 4"]
# plot_weights_over_time(rec_weights_all[0],
#                        titles=  labs,
#                        fname="{}/Rec_w.svg".format(op_plots_folder),
#                        cmaps='gray_r')

# plot_weights_over_time(rec_ctx_weights_all[0],
#                        titles=  labs,
#                        fname="{}/Rec_w_ctx.svg".format(op_plots_folder),
#                        cmaps='gray_r')

labs = ["Day 0"] + [f"Day {i+1}" for i in range(N_off_days)]
plot_weights_over_time(rec_weights_all[0],
                       titles=  labs,
                       fname="{}/Rec_w.svg".format(op_plots_folder),
                       cmaps='gray_r')

plot_weights_over_time(rec_ctx_weights_all[0],
                       titles=  labs,
                       fname="{}/Rec_w_ctx.svg".format(op_plots_folder),
                       cmaps='gray_r')
