import numpy as np
from plotting_widget import *
import matplotlib.pyplot as plt
import json

fformat = ".png"

def PlotAll(input_data_folder="./data/CNT_fast_drift_with_limited1_IP_lowI", op_plot_folder="./plots/CNT_fast_drift_with_limited1_IP_lowI"):
    # input_data_folder = "./data/CNT_fast_drift_wo_IP_lowI"
    # op_plot_folder = "./plots/CNT_fast_drift_wo_IP_lowI"
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
    seqA = sim_params["t_series"]
    E_fl_ctx = sim_params["E_fl_ctx"]
    threshold = 2#sim_params["threshold"]
    ID = sim_params["ID"]
    N_off_days = sim_params["N_off_days"]
    off_days = sim_params["off_days"]
    E_mod = sim_params["E_mod"]
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
    last_activity_ctx_all = np.load("{}/last_activity_ctx.npy".format(input_data_folder)) # shape: (sims, time, neurons)
    FR_history_all = np.load("{}/FR_history.npy".format(input_data_folder)) # shape: (sims, time, neurons)
    EX_history_all = np.load("{}/EX_history.npy".format(input_data_folder)) # shape:
    FR_op_history_all = np.load("{}/FR_history_op.npy".format(input_data_folder))
    last_activity_all_ctx = np.load("{}/last_activity_ctx.npy".format(input_data_folder)) # shape: (sims, time, neurons)
    FR_history_all_ctx = np.load("{}/FR_history_ctx.npy".format(input_data_folder)) # shape: (sims, time, neurons)
    EX_history_all_ctx = np.load("{}/EX_history_ctx.npy".format(input_data_folder)) # shape: 
    rec_weights_all = np.load("{}/rec_weights.npy".format(input_data_folder))
    rec_ctx_weights_all = np.load("{}/rec_ctx_weights.npy".format(input_data_folder))
    mtl_op_weights_all = np.load("{}/mtl_op_weights.npy".format(input_data_folder))
    ctx_op_weights_all = np.load("{}/ctx_op_weights.npy".format(input_data_folder))
    mtl_ctx_weights_all = np.load("{}/mtl_ctx_weights.npy".format(input_data_folder))
    ctx_mtl_weights_all = np.load("{}/ctx_mtl_weights.npy".format(input_data_folder))
    input_history = np.load("{}/input_history.npy".format(input_data_folder),allow_pickle=True)
    FR_history_th = (FR_history_all > threshold).astype(float)*FR_history_all
    FR_history_th_ctx = (FR_history_all_ctx > threshold).astype(float)*FR_history_all_ctx
    last_activity_th = (last_activity_all > threshold).astype(float)*last_activity_all
    last_activity_th_ctx = (last_activity_all_ctx > threshold).astype(float)*last_activity_all_ctx
    plot_engram_size(last_activity_all, threshold=threshold, title = "Engram size (HPC)",fname="{}/engram_size_HPC{}".format(op_plot_folder,fformat))
    plot_engram_size(last_activity_all_ctx, threshold=threshold,title = "Engram size (CTX)", fname="{}/engram_size_CTX{}".format(op_plot_folder,fformat))

    # breakpoint()
    # total_time = 22000
    plot_corr_matrix(last_activity_all[0], fname="{}/corr_matrix{}".format(op_plot_folder,fformat))
    plot_corr_matrix(last_activity_ctx_all[0], fname="{}/corr_matrix_ctx{}".format(op_plot_folder,fformat))
    timepoints = np.arange(0,total_time,1)*1
    plot_firing_rate(timepoints, FR_op_history_all[:, :, 0],lab = "Output neuron",
                    xlabel="Time (s)", ylabel="Firing Rate (Hz)", c="r",fname= "{}/OP_neuron_activity{}".format(op_plot_folder,fformat))
    # breakpoint()

    plot_activity_n_excitability_time([FR_history_th[-1].T,FR_history_th_ctx[-1].T],
                        titles=['Neuronal Activity (HPC)',
                                    'Neuronal Activity (CTX)'],
                        seqA=seqA,
                        fname="{}/Activity{}".format(op_plot_folder,fformat),
                        cmaps=['Oranges', 'Greens'])


    plot_activity_n_excitability_time([EX_history_all[-1].T,EX_history_all_ctx[-1].T,input_history[-1].T],
                        titles=['Neuronal Excitability (HPC)',
                                    'Neuronal Excitability (CTX)',
                                    'Input Activity'],
                        seqA=seqA,
                        fname="{}/Excitability{}".format(op_plot_folder,fformat),
                        cmaps=['Blues', 'Greens','Reds'])
    # labs = ["FC"] t [f"Off {it1}" for i in range(N_off_days)]
    # plot_weights_over_time(rec_weights_all[0],
    #                        titles=  labs,
    #                        fname="./plots/Reimagined/Rec_w{}",
    #                        cmap='gray_r')

    cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
    xlabs = [f"{i}" for i in range(N_off_days)]
    Title = "Ensemble similarity"
    # plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr{}", use_bar_plot=True)
    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_all ,                # shape: (sims, time, neurons)
        ref_time_idx=0,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/encoding_vs_others_mean_std{}".format(op_plot_folder,fformat),
        cmap = "Oranges",
        marker = "^"
    )
    mean_corr_ctx, std_corr_ctx, per_sim_corr_ctx, idx_ctx = plot_mean_std_corr_over_time(
        last_activity_ctx_all ,                # shape: (sims, time, neurons)
        ref_time_idx=0,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/encoding_vs_others_mean_std_ctx{}".format(op_plot_folder,fformat),
        cmap = "Greens"
    )
    mean_DR = np.sum(1-mean_corr)/(N_off_days)
    mean_DR_ctx = np.sum(1-mean_corr_ctx)/(N_off_days)
    print("excitability boosts:", E_fl, E_fl_ctx)
    print("Normalized drift rate:", mean_DR)
    print("Normalized drift rate:", mean_DR_ctx)

    xlabs = [f"{i-dop}" for i in range(N_off_days)]
    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_all ,                # shape: (sims, time, neurons)
        ref_time_idx=dop,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/off1_vs_others_mean_std{}".format(op_plot_folder,fformat),
        cmap = "Oranges",
        marker = "^"
    )

    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_ctx_all ,                # shape: (sims, time, neurons)
        ref_time_idx=dop,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/off1_vs_others_mean_std_ctx{}".format(op_plot_folder,fformat),
        cmap = "Greens"
    )

    xlabs = [f"{i}" for i in range(N_off_days)]
    Title = "Ensemble similarity"
    # plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined//Recall_corr{}", use_bar_plot=True)
    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_all ,                # shape: (sims, time, neurons)
        ref_time_idx=-1,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/recall_vs_others_mean_std{}".format(op_plot_folder,fformat),
        cmap = "Oranges",
        marker = "^"

    )

    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_ctx_all ,                # shape: (sims, time, neurons)
        ref_time_idx=-1,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/recall_vs_others_mean_std_ctx{}".format(op_plot_folder,fformat),
        cmap = "Greens",
        

    )

    # cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
    # xlabs = ["Off 1","Off 2","Off 3"]
    # Title = "Ensemble similarity"
    # # breakpoint()
    # # plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr{}", use_bar_plot=True)
    # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    #     last_activity_all[:,:-1,:] ,                # shape: (sims, time, neurons)
    #     ref_time_idx=0,         # Encoding
    #     xlabels=xlabs,         # must match number of non-ref times
    #     include_ref_bar=False,
    #     title="Cell population \n activity correlation",
    #     fname="{}/encoding_vs_offline_mean_std{}".format(op_plot_folder,fformat),
    #     cmap = "Oranges",
    #     marker = "^"
    # )
    # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    #     last_activity_all_ctx[:,:-1,:] ,                # shape: (sims, time, neurons)
    #     ref_time_idx=0,         # Encoding
    #     xlabels=xlabs,         # must match number of non-ref times
    #     include_ref_bar=False,
    #     title="Cell population \n activity correlation",
    #     fname="{}/encoding_vs_offline_mean_std_ctx{}".format(op_plot_folder,fformat),
    #     cmap = "Greens"
    # )

    # xlabs = [f"Off {i+1}" for i in range(N_off_days)]
    # # Title = "Ensemble similarity"
    # # # plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined//Recall_corr{}", use_bar_plot=True)
    # # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    # #     last_activity_all[:,1:,:],                # shape: (sims, time, neurons)
    # #     ref_time_idx=-1,         # Encoding
    # #     xlabels=xlabs,         # must match number of non-ref times
    # #     include_ref_bar=False,
    # #     title="Cell population \n activity correlation",
    # #     fname="{}/recall_vs_offline_mean_std{}".format(op_plot_folder,fformat),
    # #     cmap = "Oranges",
    # #     marker = "^"

    # # )

    # # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    # #     last_activity_all_ctx[:,1:,:],                # shape: (sims, time, neurons)
    # #     ref_time_idx=-1,         # Encoding
    # #     xlabels=xlabs,         # must match number of non-ref times
    # #     include_ref_bar=False,
    # #     title="Cell population \n activity correlation",
    # #     fname="{}/recall_vs_offline_mean_std_ctx{}".format(op_plot_folder,fformat),
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
    # plt.savefig("{}/overlap_frac_first_mean_sem{}".format(op_plot_folder,fformat))
    # plt.show()


    # x1, y1, r1 = plot_first_activity_vs_active_sessions(
    #     last_activity_all, threshold=threshold,
    #     first_session_idx=0,
    #     mode="concat",
    #     include_ref_in_count=True,
    #     fname="{}/first_activity_vs_counts_concat{}".format(op_plot_folder,fformat)
    # )
    # print(r1)

    # counts_x, mean_y, sem_y, n = plot_sessions_count_vs_activity_sem(
    #     last_activity_all, threshold=threshold,
    #     first_session_idx=0,
    #     include_ref_in_count=True,
    #     sem_mode='pooled',
    #     fname="{}/first_act_vs_sessions_pooled{}".format(op_plot_folder,fformat)
    # )
    # labs = ["Encoding","Day 1","Day 2","Day 3","Day 4"]
    # plot_weights_over_time(rec_weights_all[0],
    #                        titles=  labs,
    #                        fname="{}/Rec_w{}".format(op_plot_folder,fformat),
    #                        cmaps='gray_r')

    # plot_weights_over_time(rec_ctx_weights_all[0],
    #                        titles=  labs,
    #                        fname="{}/Rec_w_ctx{}".format(op_plot_folder,fformat),
    #                        cmaps='gray_r')

    labs = [f"Day {i+1}" for i in off_days]
    plot_weights_over_time(rec_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/Rec_w{}".format(op_plot_folder,fformat),
                        cmaps='gray_r')

    plot_weights_over_time(rec_ctx_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/Rec_w_ctx{}".format(op_plot_folder,fformat),
                        cmaps='gray_r')

    plot_weights_over_time(mtl_op_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/mtl_op_w{}".format(op_plot_folder,fformat),
                        cmaps='gray_r')
    plot_weights_over_time(ctx_op_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/ctx_op_w{}".format(op_plot_folder,fformat),
                        cmaps='gray_r')
    plot_weights_over_time(mtl_ctx_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/mtl_ctx_w{}".format(op_plot_folder,fformat),
                        cmaps='gray_r')
    plot_weights_over_time(ctx_mtl_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/ctx_mtl_w{}".format(op_plot_folder,fformat),
                        cmaps='gray_r')



def PlotAll3R(input_data_folder="../data/3R_CNT_fast_drift_with_limited7_IP_lowI", op_plot_folder="../plots/3R_CNT_fast_drift_with_limited7_IP_lowI"):
    # input_data_folder = "./data/CNT_fast_drift_wo_IP_lowI"
    # op_plot_folder = "./plots/CNT_fast_drift_wo_IP_lowI"
    # Path to your JSON file
    json_path = "{}/all_params.json".format(input_data_folder)
    dop = 2
    with open(json_path, "r") as f:
        sim_params = json.load(f)

    # breakpoint()
    sim_params = sim_params["simulation_params"]
    # Explicitly assign variables
    seqA = sim_params["t_series"]
    # E_fl_acc = sim_params["E_fl_acc"]
    threshold = 2#sim_params["threshold"]
    ID = sim_params["ID"]
    N_off_days = sim_params["N_off_days"]
    off_days = sim_params["off_days"]
    IR = sim_params["IR"]
    total_time = sim_params["total_time"]
    
    import numpy as np

    FR_HPC_history_all = np.load(f"{input_data_folder}/FR_HPC_history_all.npy", allow_pickle=True)
    FR_RSC_history_all = np.load(f"{input_data_folder}/FR_RSC_history_all.npy", allow_pickle=True)
    FR_ACC_history_all = np.load(f"{input_data_folder}/FR_ACC_history_all.npy", allow_pickle=True)
    FR_op_history_all = np.load(f"{input_data_folder}/FR_op_history_all.npy", allow_pickle=True)

    EX_HPC_history_all = np.load(f"{input_data_folder}/EX_HPC_history_all.npy", allow_pickle=True)
    EX_RSC_history_all = np.load(f"{input_data_folder}/EX_RSC_history_all.npy", allow_pickle=True)
    EX_ACC_history_all = np.load(f"{input_data_folder}/EX_ACC_history_all.npy", allow_pickle=True)

    last_activity_HPC_all = np.load(f"{input_data_folder}/last_activity_HPC_all.npy", allow_pickle=True)
    last_activity_RSC_all = np.load(f"{input_data_folder}/last_activity_RSC_all.npy", allow_pickle=True)
    last_activity_ACC_all = np.load(f"{input_data_folder}/last_activity_ACC_all.npy", allow_pickle=True)

    input_history_all = np.load(f"{input_data_folder}/input_history_all.npy", allow_pickle=True)
    rec_HPC_weights_all = np.load(f"{input_data_folder}/rec_HPC_weights_all.npy", allow_pickle=True)
    rec_RSC_weights_all = np.load(f"{input_data_folder}/rec_RSC_weights_all.npy", allow_pickle=True)
    rec_ACC_weights_all = np.load(f"{input_data_folder}/rec_ACC_weights_all.npy", allow_pickle=True)
    HPC_RSC_weights_all = np.load(f"{input_data_folder}/HPC_RSC_weights_all.npy", allow_pickle=True)
    RSC_ACC_weights_all = np.load(f"{input_data_folder}/RSC_ACC_weights_all.npy", allow_pickle=True)
    HPC_ACC_weights_all = np.load(f"{input_data_folder}/HPC_ACC_weights_all.npy", allow_pickle=True)

    # HPC_OP_weights_all = np.load(f"{input_data_folder}/HPC_OP_weights.npy", allow_pickle=True)
    ACC_OP_weights_all = np.load(f"{input_data_folder}/ACC_OP_weights_all.npy", allow_pickle=True)
    ACC_HPC_weights_all = np.load(f"{input_data_folder}/ACC_HPC_weights_all.npy", allow_pickle=True)
        
    
    FR_history_th_hpc = (FR_HPC_history_all > threshold).astype(float)*FR_HPC_history_all
    FR_history_th_acc = (FR_ACC_history_all > threshold).astype(float)*FR_ACC_history_all
    FR_history_th_rsc = (FR_RSC_history_all > threshold).astype(float)*FR_RSC_history_all
    
    last_activity_HPC_all_th = (last_activity_HPC_all > threshold).astype(float)*last_activity_HPC_all
    last_activity_RSC_all_th = (last_activity_RSC_all > threshold).astype(float)*last_activity_RSC_all
    last_activity_ACC_all_th = (last_activity_ACC_all > threshold).astype(float)*last_activity_ACC_all
    
    plot_engram_size(last_activity_HPC_all, threshold=threshold, title = "Engram size (HPC)",fname="{}/engram_size_HPC{}".format(op_plot_folder,fformat))
    plot_engram_size(last_activity_RSC_all, threshold=threshold,title = "Engram size (RSC)", fname="{}/engram_size_CTX{}".format(op_plot_folder,fformat))
    plot_engram_size(last_activity_ACC_all, threshold=threshold,title = "Engram size (ACC)", fname="{}/engram_size_CTX{}".format(op_plot_folder,fformat))

    plot_corr_matrix(last_activity_HPC_all_th[0], fname="{}/corr_matrix_hpc{}".format(op_plot_folder,fformat))
    plot_corr_matrix(last_activity_RSC_all_th[0], fname="{}/corr_matrix_rsc{}".format(op_plot_folder,fformat))
    plot_corr_matrix(last_activity_ACC_all_th[0], fname="{}/corr_matrix_acc{}".format(op_plot_folder,fformat))
    
    timepoints = np.arange(0,total_time,1)*1
    plot_firing_rate(timepoints, FR_op_history_all[:, :, 0],lab = "Output neuron",
                    xlabel="Time (s)", ylabel="Firing Rate (Hz)", c="r",fname= "{}/OP_neuron_activity{}".format(op_plot_folder,fformat))
    # breakpoint()

    plot_activity_n_excitability_time([FR_history_th_hpc[-1].T,FR_history_th_rsc[-1].T,FR_history_th_acc[-1].T],
                        titles=['Neuronal Activity (HPC)',
                                'Neuronal Activity (RSC)',
                                'Neuronal Activity (ACC)'],
                        seqA=seqA,
                        fname="{}/Activity{}".format(op_plot_folder,fformat),
                        cmaps=['Oranges', 'Greens',"Blues"])


    plot_activity_n_excitability_time([EX_HPC_history_all[-1].T,EX_RSC_history_all[-1].T,EX_ACC_history_all[-1].T],
                        titles=['Neuronal Excitability (HPC)',
                                'Neuronal Excitability (RSC)',
                                'Neuronal Excitability (ACC)'],
                        seqA=seqA,
                        fname="{}/Excitability{}".format(op_plot_folder,fformat),
                        cmaps=['Greens', 'Greens','Greens'])
    # labs = ["FC"] t [f"Off {it1}" for i in range(N_off_days)]
    # plot_weights_over_time(rec_weights_all[0],
    #                        titles=  labs,
    #                        fname="./plots/Reimagined/Rec_w{}",
    #                        cmap='gray_r')

    cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
    xlabs = [f"{i}" for i in range(N_off_days)]
    Title = "Ensemble similarity"
    # plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr{}", use_bar_plot=True)
    mean_corr_hpc, std_corr_hpc, per_sim_corr_hpc, idx_hpc = plot_mean_std_corr_over_time(
        last_activity_ACC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=0,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/encoding_vs_others_mean_std_hpc{}".format(op_plot_folder,fformat),
        cmap = "Oranges",
        marker = "^"
    )
    mean_corr_rsc, std_corr_rsc, per_sim_corr_rsc, idx_rsc = plot_mean_std_corr_over_time(
        last_activity_RSC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=0,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/encoding_vs_others_mean_std_rsc{}".format(op_plot_folder,fformat),
        cmap = "Blues"
    )
    mean_corr_acc, std_corr_acc, per_sim_corr_acc, idx_acc = plot_mean_std_corr_over_time(
        last_activity_ACC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=0,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/encoding_vs_others_mean_std_acc{}".format(op_plot_folder,fformat),
        cmap = "Greens"
    )
    mean_DR_hpc = np.sum(1-mean_corr_hpc)/(N_off_days)
    mean_DR_rsc = np.sum(1-mean_corr_rsc)/(N_off_days)
    mean_DR_acc = np.sum(1-mean_corr_acc)/(N_off_days)
    # print("excitability boosts:", E_fl, E_fl_acc)
    print("Normalized drift rate (HPC):", mean_DR_hpc)
    print("Normalized drift rate (RSC):", mean_DR_rsc)
    print("Normalized drift rate (ACC):", mean_DR_acc)

    xlabs = [f"{i-dop}" for i in range(N_off_days)]
    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_HPC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=dop,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/off1_vs_others_mean_std_hpc{}".format(op_plot_folder,fformat),
        cmap = "Oranges",
        marker = "^"
    )

    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_ACC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=dop,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/off1_vs_others_mean_std_acc{}".format(op_plot_folder,fformat),
        cmap = "Greens"
    )

    xlabs = [f"{i}" for i in range(N_off_days)]
    Title = "Ensemble similarity"
    # plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined//Recall_corr{}", use_bar_plot=True)
    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_HPC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=-1,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/recall_vs_others_mean_std_hpc{}".format(op_plot_folder,fformat),
        cmap = "Oranges",
        marker = "^"

    )

    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_ACC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=-1,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/recall_vs_others_mean_std_acc{}".format(op_plot_folder,fformat),
        cmap = "Greens",
        

    )

    # cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
    # xlabs = ["Off 1","Off 2","Off 3"]
    # Title = "Ensemble similarity"
    # # breakpoint()
    # # plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr{}", use_bar_plot=True)
    # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    #     last_activity_all[:,:-1,:] ,                # shape: (sims, time, neurons)
    #     ref_time_idx=0,         # Encoding
    #     xlabels=xlabs,         # must match number of non-ref times
    #     include_ref_bar=False,
    #     title="Cell population \n activity correlation",
    #     fname="{}/encoding_vs_offline_mean_std{}".format(op_plot_folder,fformat),
    #     cmap = "Oranges",
    #     marker = "^"
    # )
    # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    #     last_activity_all_acc[:,:-1,:] ,                # shape: (sims, time, neurons)
    #     ref_time_idx=0,         # Encoding
    #     xlabels=xlabs,         # must match number of non-ref times
    #     include_ref_bar=False,
    #     title="Cell population \n activity correlation",
    #     fname="{}/encoding_vs_offline_mean_std_acc{}".format(op_plot_folder,fformat),
    #     cmap = "Greens"
    # )

    # xlabs = [f"Off {i+1}" for i in range(N_off_days)]
    # # Title = "Ensemble similarity"
    # # # plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined//Recall_corr{}", use_bar_plot=True)
    # # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    # #     last_activity_all[:,1:,:],                # shape: (sims, time, neurons)
    # #     ref_time_idx=-1,         # Encoding
    # #     xlabels=xlabs,         # must match number of non-ref times
    # #     include_ref_bar=False,
    # #     title="Cell population \n activity correlation",
    # #     fname="{}/recall_vs_offline_mean_std{}".format(op_plot_folder,fformat),
    # #     cmap = "Oranges",
    # #     marker = "^"

    # # )

    # # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    # #     last_activity_all_acc[:,1:,:],                # shape: (sims, time, neurons)
    # #     ref_time_idx=-1,         # Encoding
    # #     xlabels=xlabs,         # must match number of non-ref times
    # #     include_ref_bar=False,
    # #     title="Cell population \n activity correlation",
    # #     fname="{}/recall_vs_offline_mean_std_acc{}".format(op_plot_folder,fformat),
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
    # plt.savefig("{}/overlap_frac_first_mean_sem{}".format(op_plot_folder,fformat))
    # plt.show()


    # x1, y1, r1 = plot_first_activity_vs_active_sessions(
    #     last_activity_all, threshold=threshold,
    #     first_session_idx=0,
    #     mode="concat",
    #     include_ref_in_count=True,
    #     fname="{}/first_activity_vs_counts_concat{}".format(op_plot_folder,fformat)
    # )
    # print(r1)

    # counts_x, mean_y, sem_y, n = plot_sessions_count_vs_activity_sem(
    #     last_activity_all, threshold=threshold,
    #     first_session_idx=0,
    #     include_ref_in_count=True,
    #     sem_mode='pooled',
    #     fname="{}/first_act_vs_sessions_pooled{}".format(op_plot_folder,fformat)
    # )
    # labs = ["Encoding","Day 1","Day 2","Day 3","Day 4"]
    # plot_weights_over_time(rec_weights_all[0],
    #                        titles=  labs,
    #                        fname="{}/Rec_w{}".format(op_plot_folder,fformat),
    #                        cmaps='gray_r')

    # plot_weights_over_time(rec_acc_weights_all[0],
    #                        titles=  labs,
    #                        fname="{}/Rec_w_acc{}".format(op_plot_folder,fformat),
    #                        cmaps='gray_r')

    labs = [f"Day {i+1}" for i in off_days]
    plot_weights_over_time(rec_HPC_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/Rec_w_hpc{}".format(op_plot_folder,fformat),
                        cmaps='gray_r')
    plot_weights_over_time(rec_RSC_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/Rec_w_rsc{}".format(op_plot_folder,fformat),
                        cmaps='gray_r')
    plot_weights_over_time(rec_ACC_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/Rec_w_acc{}".format(op_plot_folder,fformat),
                        cmaps='gray_r')

    # plot_weights_over_time(mtl_op_weights_all[-1,off_days],
    #                     titles=  labs,
    #                     fname="{}/mtl_op_w{}".format(op_plot_folder,fformat),
    #                     cmaps='gray_r')
    plot_weights_over_time(ACC_OP_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/acc_op_w{}".format(op_plot_folder,fformat),
                        cmaps='gray_r')
    
    plot_weights_over_time(HPC_RSC_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/hpc_rsc_w{}".format(op_plot_folder,fformat),
                        cmaps='gray_r')
    plot_weights_over_time(RSC_ACC_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/rsc_acc_w{}".format(op_plot_folder,fformat),
                        cmaps='gray_r')
    plot_weights_over_time(HPC_ACC_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/hpc_acc_w{}".format(op_plot_folder,fformat),
                        cmaps='gray_r')


if __name__ == "__main__":
    PlotAll3R()