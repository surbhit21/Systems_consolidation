import numpy as np
import matplotlib.pyplot as plt
import torch
from plotting_widget import plot_activity_n_excitability_time, plot_weights_over_time, save_plot
# from EI_Hebb_Vogels import EI_Hebb_Vogels
inp_folder_data = "./data/HPC_W_Norm_noextc"
# op_plots_folder_plots = "./plots/HPC_W_Norm"
data = np.load("{}/EI_HPC_with_normalization.npz".format(inp_folder_data), allow_pickle=True)

MTL_weights_ee=data['MTL_weights_ee']
MTL_weights_ei=data['MTL_weights_ei']
MTL_weights_ie=data['MTL_weights_ie']
MTL_weights_ii=data['MTL_weights_ii']
STIM_MTL_weights_e= data['STIM_MTL_weights_e']
STIM_MTL_weights_i= data['STIM_MTL_weights_i']
E_rate= data['E_rate']
I_rate= data['I_rate']
ext_MTL_e= data['ext_MTL_e']
ext_MTL_i= data['ext_MTL_i']
op_plots_folder = "./plots/HPC_W_Norm"
labels = ["Before" "After"]
Ne = 20
Ni = 20
Ninp = 20
pref_orientations = torch.linspace(0, 180, Ninp, device="cpu")
fig, axes = plt.subplots(2, 2, figsize=(10, 4), sharey=True)
# labels = ["Before", "During", "After"]

    # --- Plot excitatory ---
for i, w in enumerate(STIM_MTL_weights_e):
    for widx, we in enumerate(w):
        axes[0][i].plot(pref_orientations,we,alpha=(1/20)*widx,color='b')
    # axes[0][i].set_title("Excitatory neurons (E)")
    # axes[0].set_xlabel("Neuron index (sorted)")
    # axes[0].set_ylabel("Feedforward sum")
    # axes[0].grid(True)
    # axes[0].legend()

# --- Plot inhibitory ---
for i, w in enumerate(STIM_MTL_weights_i):
    for wdx, wi in enumerate(w):
        axes[1][i].plot(pref_orientations,wi,alpha=(1/20)*widx,color='r')
    # axes[1].set_title("Inhibitory neurons (I)")
    # axes[1].set_xlabel("Neuron index (sorted)")
    # axes[1].set_ylabel("Feedforward sum")
    # axes[1].grid(True)
    # axes[1].legend()
plt.suptitle("Evolution of feedforward input sums during training")
plt.tight_layout()
plt.savefig("{}/FF_weights_evolution.png".format(op_plots_folder))
plt.show()

t_s = np.arange(E_rate.shape[0]) * 0.01
plt.plot(t_s,E_rate.mean(axis=1), label='E')
plt.plot(t_s,I_rate.mean(axis=1), label='I')
plt.xlabel('Time (a.u.)')
plt.ylabel('Average firing rate (Hz)')
plt.legend()
plt.title('Average Firing Rates')
save_plot("{}/AVG_FR.png".format(op_plots_folder))
plt.show()
#
# breakpoint()
pre_labs = ["T = 0","After Encoding", "OF1"]
plot_activity_n_excitability_time([ext_MTL_e.T,ext_MTL_i.T],
                    titles=['Neuronal Activity (Excitatory)','Neuronal Activity (Inhibitory)'],
                    fname="{}/Activity.png".format(op_plots_folder),
                    cmaps=['Greens','Blues'])
labs = ["E -> E: "+ x for x in pre_labs]
plot_weights_over_time(MTL_weights_ee,
                    titles= labs,
                    fname="{}/Rec_w_MT_ee.png".format(op_plots_folder),
                    cmaps='gray_r')
labs = ["I -> E: "+ x for x in pre_labs]
plot_weights_over_time(MTL_weights_ei,
                    titles= labs,
                    fname="{}/Rec_w_MT_ei.png".format(op_plots_folder),
                    cmaps='gray_r')
labs = ["E -> I: "+ x for x in pre_labs]
plot_weights_over_time(MTL_weights_ie,
                    titles= labs,
                    fname="{}/Rec_w_MT_ie.png".format(op_plots_folder),
                    cmaps='gray_r')
labs = ["I -> I: "+ x for x in pre_labs]
plot_weights_over_time(MTL_weights_ii,
                    titles= labs,
                    fname="{}/Rec_w_MT_ii.png".format(op_plots_folder),
                    cmaps='gray_r')
breakpoint()
labs = ["S -> E: "+ x for x in pre_labs]
plot_weights_over_time(STIM_MTL_weights_e,
                    titles= labs,
                    fname="{}/FF_STIM_E.png".format(op_plots_folder),
                    cmaps='gray_r')
labs = ["S -> I: "+ x for x in pre_labs]
plot_weights_over_time(STIM_MTL_weights_i,
                    titles= labs,
                    fname="{}/FF_STIM_I.png".format(op_plots_folder),
                    cmaps='gray_r')

