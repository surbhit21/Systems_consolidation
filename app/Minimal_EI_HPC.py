import matplotlib.pyplot as plt
import numpy as np
import os
from plotting_widget import *
from tqdm import tqdm
import torch 
import torch.nn.functional as F
from Utilities import average_firing_rates_with_active,ensamble_overlap
import torch
from torchvision import datasets, transforms
from tqdm import trange
torch.manual_seed(2025)
def get_linearized_mnist_digit3(device="cpu", train=True, idx=0, binarize=False):
    """
    Returns a single linearized MNIST '3' as float tensor of shape [784] in [0,1].
    Picks the idx-th '3' from split (train/test).
    """
    tfms = transforms.Compose([
        transforms.ToTensor(),  # -> [1,28,28] in [0,1]
    ])
    ds = datasets.MNIST(root="./data", train=train, download=True, transform=tfms)
    threes = [(i, (y == 3)) for i, (_, y) in enumerate(ds)]
    three_indices = [i for i, ok in threes if ok]
    if len(three_indices) == 0:
        raise RuntimeError("No digit '3' found in MNIST split.")
    i = three_indices[idx % len(three_indices)]
    x, y = ds[i]  # x: [1,28,28]
    x = x.squeeze(0)  # [28,28]
    if binarize:
        x = (x > 0.5).float()
    # plt.imshow(x.cpu(), cmap='gray')
    # plt.axis('off')
    # plt.show()  
    x = x.reshape(-1).to(device)  # [784]
    return x  # [784] in [0,1]

class TwoAreaEIModel:
    def __init__(self, 
                 nMTL_e,               # number of E neurons
                 nMTL_i,               # number of I neurons
                 eMTL_e, eMTL_i,       # tonic excitability drives (shape [nMTL_e], [nMTL_i])
                 n_inp=784,                    # stimulus dimension (MNIST=28*28)
                 tau_MTL=50.0,
                 lr_e_MTL=1e-3, decay_e_MTL=5e-4,
                 lr_i_MTL=1e-3, decay_i_MTL=5e-4,
                 threshold=0.0, dt=1.0, act=torch.relu,
                 device="cpu"
                 ):
        self.device = device

        # Inputs / sizes
       
        self.nMTL_e = nMTL_e
        self.nMTL_i = nMTL_i

        self.n_inp  = n_inp

        # Dynamics
        self.tau_MTL = tau_MTL
        self.dt = dt
        self.act = act
        self.threshold = threshold

        # Learning rates/decay
        self.lr_e_MTL = lr_e_MTL
        self.decay_e_MTL = decay_e_MTL
        self.lr_i_MTL = lr_i_MTL
        self.decay_i_MTL = decay_i_MTL

        # External tonic drives
        self.ext_MTL_e = eMTL_e.to(device)
        self.ext_MTL_i = eMTL_i.to(device)

        # States
        self.r_MTL_e = torch.zeros(nMTL_e, device=device)
        self.r_MTL_i = torch.zeros(nMTL_i, device=device)

        # Target rates for inhibitory plasticity
        self.rho_MTL_e = 2*torch.ones(nMTL_e, device=device)
        
        # Recurrent weights (E/I blocks) — shapes corrected
        # MTL
        mu, std = 0.1, 0.05
        self.W_MTL_MTL_ee = torch.zeros(nMTL_e, nMTL_e, device=device)   # E←E
        self.W_MTL_MTL_ei = torch.zeros(nMTL_e, nMTL_i, device=device)   # E←I
        self.W_MTL_MTL_ie = torch.zeros(nMTL_i, nMTL_e, device=device)   # I←E
        self.W_MTL_MTL_ii = torch.zeros(nMTL_i, nMTL_i, device=device) #torch.zeros(torch.randn(nMTL_i, nMTL_i, device=device)) / torch.sqrt(torch.tensor(nMTL_i, dtype=torch.float, device=device))  # I←I

        # Target sums for weight normalization 
        self.W_EE = float(10)  # target sum of EE weights onto each E neuron
        self.W_EI = None # target sum of EI weights onto each E neuron
        self.W_IE = None  # target sum of IE weights onto eacch E neuron
        self.W_II = None # target sum of IE weights onto each E neuron

        # Stimulus feedforward weights (stim -> E)
        # Small random initial fan-in
        scale_MTL = 1.0 / (n_inp ** 0.5)
        self.W_STIM_MTL_ef = torch.normal(mu,std,size=(nMTL_e,n_inp)) * scale_MTL
        self.W_STIM_MTL_if = torch.normal(mu,std,size=(nMTL_i,n_inp)) * scale_MTL
        # self.W_STIM_MTL_ef = self.W_STIM_MTL_ef/ (self.W_STIM_MTL_ef.sum(dim=1, keepdim=True)) 
        # self.W_STIM_MTL_if = self.W_STIM_MTL_if/ (self.W_STIM_MTL_if.sum(dim=1, keepdim=True))   
        # breakpoint()
        self.W_EF = None  # target sum of feedforward weights onto each E neuron
        self.W_IF = None  # target sum of feedforward weights onto each I neuron

        # Current stimulus vector [n_inp]; default zeros
        self.stim = torch.zeros(n_inp, device=device)

    @torch.no_grad()
    
    def set_stimulus(self, stim_vec,  gain=1.0):
        """
        stim_vec: tensor-like [n_inp]. Typically a flattened MNIST image.
        """
        s1 = torch.as_tensor(stim_vec, device=self.device, dtype=torch.float32).reshape(-1)
        if s1.numel() != self.n_inp:
            raise ValueError(f"stim size {s1.numel()} != n_inp {self.n_inp}")
        
        s1 = torch.clamp(s1, 0.0, 1.0)
        s1 = (s1 > 0.5).float()
        self.stim = gain * s1
        x = self.stim.reshape(28, 28)
        # breakpoint()
        # print(x)
        # plt.imshow(x.cpu(), cmap='gray')
        # plt.axis('off')
        # plt.show() 
    
    def step(self,plastic=True,normalize=True):
        """
        One Euler step of rate dynamics.
        """
        # Inputs to MTL
        total_inp_MTL_e = (
            self.W_MTL_MTL_ee @ self.r_MTL_e          # E←E
            - self.W_MTL_MTL_ei @ self.r_MTL_i        # E←I (note W_ei is positive, so minus sign here)
            + self.W_STIM_MTL_ef @ self.stim           # E←Stim
            + self.ext_MTL_e                          # tonic excitability drive  
        )
        total_inp_MTL_i = (
            self.W_MTL_MTL_ie @ self.r_MTL_e          # I←E
            - self.W_MTL_MTL_ii @ self.r_MTL_i        # I←I (note W_ii is positive, so minus sign here)
            + self.W_STIM_MTL_if @ self.stim           # I←Stim
            + self.ext_MTL_i                          # tonic excitability drive    
        )



        # Rate updates (rectified)
        dr_MTL_e_dt = (-self.r_MTL_e + self.act(total_inp_MTL_e )) / self.tau_MTL
        dr_MTL_i_dt = (-self.r_MTL_i + self.act(total_inp_MTL_i )) / self.tau_MTL
    
        
        self.r_MTL_e += self.dt * dr_MTL_e_dt
        self.r_MTL_i += self.dt * dr_MTL_i_dt

        if plastic:
            self.W_MTL_MTL_ee += self.dt * (self.lr_e_MTL * torch.outer(self.r_MTL_e, self.r_MTL_e))
            self.W_MTL_MTL_ie += self.dt * (self.lr_i_MTL * torch.outer(self.r_MTL_i, self.r_MTL_e ))
            self.W_MTL_MTL_ei += self.dt * (self.lr_e_MTL * torch.outer(self.r_MTL_e, self.r_MTL_i))
            self.W_MTL_MTL_ii += self.dt * (self.lr_i_MTL * torch.outer(self.r_MTL_i, self.r_MTL_i))
            self.W_STIM_MTL_ef += self.dt * (self.lr_e_MTL * torch.outer(self.r_MTL_e, self.stim))
            self.W_STIM_MTL_if += self.dt * (self.lr_i_MTL * torch.outer(self.r_MTL_i, self.stim))
            if normalize:
                self.normalize_weights()
        return {
            "r_MTL_e": self.r_MTL_e,
            "r_MTL_i": self.r_MTL_i,
        }
    @torch.no_grad()
    def reset_states(self):
        self.r_MTL_e.zero_()
        self.r_MTL_i.zero_()

    @torch.no_grad()
    def normalize_weights(self ):
        """
        Normalize incoming weights to each neuron to fixed sum if the sum is defined (>0).
        
        This is done for all 6 weight matrices.
        """
        total_weights_e = self.W_MTL_MTL_ee.sum(dim=1, keepdim=True) + self.W_STIM_MTL_ef.sum(dim=1,keepdim=True)
        total_weights_i = self.W_MTL_MTL_ie.sum(dim=1, keepdim=True) + self.W_STIM_MTL_if.sum(dim=1,keepdim=True)

        total_EI = self.W_MTL_MTL_ei.sum(dim=1, keepdim=True)
        total_II = self.W_MTL_MTL_ii.sum(dim=1, keepdim=True)
        # breakpoint()
         # Normalize E←E and E←Stim weights to W_EE
         # Normalize I←E and I←Stim weights to W_IE
         # Normalize E←I weights to W_EI
         # Normalize I←I weights to W_II
        # breakpoint()
        # Only do this if the target sum is defined (>0) and current sum >0 to avoid NaNs
        if self.W_EE is not None:
            # mask = torch.nonzero(total_weights_e > 0).float()
            self.W_MTL_MTL_ee *= (self.W_EE / total_weights_e)
            self.W_STIM_MTL_ef *= (self.W_EE / total_weights_e)
        if self.W_IE is not None:
            self.W_MTL_MTL_ie *= (self.W_IE / total_weights_i)
            self.W_STIM_MTL_if *= (self.W_IE / total_weights_i)
        if self.W_EI is not None:
            self.W_MTL_MTL_ei *= (self.W_EI / total_EI)
        if self.W_II is not None:
            self.W_MTL_MTL_ii *= (self.W_II / total_II)

        # breakpoint()
if __name__ == "__main__":
    device = "cpu"
    # tonic inputs/excitability (shape to match pops; zeros ok)
    nMTL_e, nMTL_i = 4000, 1000
    MTL_inp = torch.zeros(nMTL_e, device=device)    
    ext_MTL_e = torch.abs(torch.randn(nMTL_e, device=device))#torch.rand(nMTL_e, device=device)
    ext_MTL_i = torch.abs(torch.randn(nMTL_i, device=device))
    model = TwoAreaEIModel(
        nMTL_e, 
        nMTL_i, 
        ext_MTL_e, 
        ext_MTL_i,
        n_inp=784,
        device=device
    )
    model.ext_MTL_e[:1000] += 3
    model.ext_MTL_i[:250] += 3

    # Load one MNIST '3' stimulus and set it

    T_total = 1000
    stim = get_linearized_mnist_digit3(device=device, train=True, idx=0, binarize=False)
    model.set_stimulus(stim, gain=5)  # gain controls drive strength


    MTL_weights_ee = [model.W_MTL_MTL_ee.cpu().detach().numpy().copy()]
    MTL_weights_ei = [model.W_MTL_MTL_ei.cpu().detach().numpy().copy()]
    MTL_weights_ie = [model.W_MTL_MTL_ie.cpu().detach().numpy().copy()]
    MTL_weights_ii = [model.W_MTL_MTL_ii.cpu().detach().numpy().copy()]
    STIM_MTL_weights_e = [model.W_STIM_MTL_ef.cpu().detach().numpy().copy()]
    STIM_MTL_weights_i = [model.W_STIM_MTL_if.cpu().detach().numpy().copy()]
    E_rate = []
    I_rate = []
    for t in trange(T_total):
        rates = model.step()
        E_rate.append(rates["r_MTL_e"].cpu().detach().numpy().copy())
        I_rate.append(rates["r_MTL_i"].cpu().detach().numpy().copy())
    

    MTL_weights_ee.append(model.W_MTL_MTL_ee.cpu().detach().numpy().copy())
    MTL_weights_ei.append(model.W_MTL_MTL_ei.cpu().detach().numpy().copy())
    MTL_weights_ie.append(model.W_MTL_MTL_ie.cpu().detach().numpy().copy())
    MTL_weights_ii.append(model.W_MTL_MTL_ii.cpu().detach().numpy().copy())
    STIM_MTL_weights_e.append(model.W_STIM_MTL_ef.cpu().detach().numpy().copy())
    STIM_MTL_weights_i.append(model.W_STIM_MTL_if.cpu().detach().numpy().copy())
    breakpoint()
    T_off = 200
    # eMTL_e[:1000] -= 3
    # eMTL_i[:250] -= 3
    # eMTL_e[1000:2000] += 3
    # eMTL_i[250:500] += 3
    # stim = get_linearized_mnist_digit3(device=device, train=True, idx=0, binarize=False)
    # model.set_stimulus(stim, gain=2)  # gain controls drive strength
    model.reset_states()
    model.stim = torch.normal(mean = 0,std = 1,size = (784,), device=device)*0.1
    for t in trange(T_off):
        rates = model.step(plastic=True)
        E_rate.append(rates["r_MTL_e"].cpu().detach().numpy().copy())
        I_rate.append(rates["r_MTL_i"].cpu().detach().numpy().copy())
    
    
    MTL_weights_ee.append(model.W_MTL_MTL_ee.cpu().detach().numpy().copy())
    MTL_weights_ei.append(model.W_MTL_MTL_ei.cpu().detach().numpy().copy())
    MTL_weights_ie.append(model.W_MTL_MTL_ie.cpu().detach().numpy().copy())
    MTL_weights_ii.append(model.W_MTL_MTL_ii.cpu().detach().numpy().copy())
    STIM_MTL_weights_e.append(model.W_STIM_MTL_ef.cpu().detach().numpy().copy())
    STIM_MTL_weights_i.append(model.W_STIM_MTL_if.cpu().detach().numpy().copy())
    



    MTL_weights_ei = np.array(MTL_weights_ei)  # [time, nE, nI]
    MTL_weights_ie = np.array(MTL_weights_ie)  # [time, nI, nE]
    MTL_weights_ee = np.array(MTL_weights_ee)  # [
    MTL_inp_weights_ii = np.array(MTL_weights_ii)  # [time, nI, nI]
    STIM_MTL_weights_e = np.array(STIM_MTL_weights_e)  #   [time, nE, nE]
    STIM_MTL_weights_i = np.array(STIM_MTL_weights_i)  # [time, nI, nI]

    E_rate = np.array(E_rate)  # [time, nE]
    I_rate = np.array(I_rate)  # [time, nI]
    # active_neurons = np.where(E_rate > theta, E_rate, 0)
    # T_total = T_total + T_off
    t_s = np.arange(E_rate.shape[0]) * model.dt
    plt.plot(t_s,E_rate.mean(axis=1), label='E')
    plt.plot(t_s,I_rate.mean(axis=1), label='I')
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Average firing rate (Hz)')
    plt.legend()
    plt.title('Average Firing Rates')
    save_plot("./plots/HPC_RNN_MNIST/AVG_FR.png")
    plt.show()
    #
    breakpoint()
    pre_labs = ["T = 0","After Encoding", "OF1"]
    plot_activity_n_excitability_time([E_rate.T,I_rate.T],
                        titles=['Neuronal Activity (Excitatory)','Neuronal Activity (Inhibitory)'],
                        fname="./plots/HPC_RNN_MNIST/Activity.png",
                        cmaps=['Greens','Blues'])
    labs = ["E -> E: "+ x for x in pre_labs]
    plot_weights_over_time(MTL_weights_ee,
                        titles= labs,
                        fname="./plots/HPC_RNN_MNIST/Rec_w_MT_ee.png",
                        cmaps='gray_r')
    labs = ["I -> E: "+ x for x in pre_labs]
    plot_weights_over_time(MTL_weights_ei,
                        titles= labs,
                        fname="./plots/HPC_RNN_MNIST/Rec_w_MT_ei.png",
                        cmaps='gray_r')
    labs = ["E -> I: "+ x for x in pre_labs]
    plot_weights_over_time(MTL_weights_ie,
                        titles= labs,
                        fname="./plots/HPC_RNN_MNIST/Rec_w_MT_ie.png",
                        cmaps='gray_r')
    labs = ["I -> I: "+ x for x in pre_labs]
    plot_weights_over_time(MTL_weights_ii,
                        titles= labs,
                        fname="./plots/HPC_RNN_MNIST/Rec_w_MT_ii.png",
                        cmaps='gray_r')
    labs = ["S -> E: "+ x for x in pre_labs]
    plot_weights_over_time(STIM_MTL_weights_e,
                        titles= labs,
                        fname="./plots/HPC_RNN_MNIST/FF_STIM_E.png",
                        cmaps='gray_r')
    labs = ["S -> I: "+ x for x in pre_labs]
    plot_weights_over_time(STIM_MTL_weights_i,
                        titles= labs,
                        fname="./plots/HPC_RNN_MNIST/FF_STIM_I.png",
                        cmaps='gray_r')

