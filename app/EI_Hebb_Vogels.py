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
                 n_inp = 784,                    # stimulus dimension (MNIST=28*28)
                 tau_e_MTL = 20.0,
                 tau_i_MTL = 10.0,
                 tau_ee_MTL = 1000,
                 tau_ei_MTL = 200,
                 tau_ii_MTL = 400, 
                 tau_ie_MTL = 600,
                 tau_ef_MTL = 5e2,
                 tau_if_MTL = 1e2,
                 threshold = 0.0, dt = 1.0, act = torch.relu,
                 device = "cpu"
                 ):
        self.device = device

        # Inputs / sizes
       
        self.nMTL_e = nMTL_e
        self.nMTL_i = nMTL_i

        self.n_inp  = n_inp

        # Dynamics
        self.tau_e_MTL = tau_e_MTL
        self.tau_i_MTL = tau_i_MTL
        self.dt = dt
        self.act = act
        self.threshold = threshold
        self.target_e_rate = 1
        self.target_i_rate = 1
        self.a = 1.0  # scaling for inhibitory plasticity
        self.b = 0

        # Learning rates/decay
        self.tau_ee_MTL = tau_ee_MTL
        self.tau_ei_MTL = tau_ei_MTL
        self.tau_ii_MTL = tau_ii_MTL
        self.tau_ie_MTL = tau_ie_MTL
        self.tau_ef_MTL = tau_ef_MTL
        self.tau_if_MTL = tau_if_MTL
        self.plast_th = 0  # plasticity threshold for excitatory plasticity
        # self.lr_ef_MTL = lr_ef_MTL

        # self.lr_if_MTL = lr_if_MTL

        # External tonic drives
        self.ext_MTL_e = eMTL_e.to(device)
        self.ext_MTL_i = eMTL_i.to(device)

        # states
        self.u_MTL_e = torch.zeros(nMTL_e, device=device)  # membrane potentials
        self.u_MTL_i = torch.zeros(nMTL_i, device=device)

        # rates = a*[u-b]_+
        self.r_MTL_e = torch.zeros(nMTL_e, device=device)
        self.r_MTL_i = torch.zeros(nMTL_i, device=device)

        # Target rates for inhibitory plasticity
        self.rho_MTL_e = 2*torch.ones(nMTL_e, device=device)
        
        # Recurrent weights (E/I blocks) — shapes corrected
        # MTL
        mu, sigma = 0,.1
        self.W_MTL_MTL_ee = torch.zeros(nMTL_e, nMTL_e)#torch.abs(torch.normal(mu,sigma,size=(nMTL_e, nMTL_e)))   # E ← E 
        self.W_MTL_MTL_ei = torch.zeros(nMTL_e, nMTL_i)#torch.abs(torch.normal(mu,sigma,size=(nMTL_e, nMTL_i)))   # E ← I  
        self.W_MTL_MTL_ie = torch.zeros(nMTL_i, nMTL_e)#torch.abs(torch.normal(mu,sigma,size=(nMTL_i, nMTL_e)))   # I ← E 
        self.W_MTL_MTL_ii = torch.zeros(nMTL_i, nMTL_i)#torch.abs(torch.normal(mu,sigma,size=(nMTL_i, nMTL_i)))   # I ← I 
        # Stimulus feedforward weights (stim -> E)
        # Small random initial fan-in
        # scale_MTL = 1.0 / (n_inp ** 0.5)
        self.W_STIM_MTL_ef = torch.abs(torch.normal(mu,sigma,size=(nMTL_e,n_inp))) 
        self.W_STIM_MTL_if = torch.abs(torch.normal(mu,sigma,size=(nMTL_i,n_inp))) 
        # self.W_STIM_MTL_ef = self.W_STIM_MTL_ef/ (self.W_STIM_MTL_ef.sum(dim=1, keepdim=True)) 
        # self.W_STIM_MTL_if = self.W_STIM_MTL_if/ (self.W_STIM_MTL_if.sum(dim=1, keepdim=True))   
        # breakpoint()
        # self.W_EF = None  # target sum of feedforward weights onto each E neuron
        # self.W_IF = None  # target sum of feedforward weights onto each I neuron

        # Current stimulus vector [n_inp]; default zeros
        self.e_stim = torch.zeros(n_inp, device=device)
        self.i_stim = torch.zeros(n_inp, device=device)

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
        self.e_stim = gain * s1
        x = self.e_stim.reshape(28, 28)
        
    
    def step(self,plastic=True):
        """
        One Euler step of rate dynamics.
        """
        # Inputs to MTL
        total_inp_MTL_e = (
            self.W_MTL_MTL_ee @ self.r_MTL_e          # E←E
            - self.W_MTL_MTL_ei @ self.r_MTL_i        # E←I (note W_ei is positive, so minus sign here)
            + self.W_STIM_MTL_ef @ self.e_stim            # E←Stim
            + self.ext_MTL_e                          # tonic excitability drive  
        )
        total_inp_MTL_i = (
            self.W_MTL_MTL_ie @ self.r_MTL_e          # I←E
            - self.W_MTL_MTL_ii @ self.r_MTL_i        # I←I (note W_ii is positive, so minus sign here)
            + self.W_STIM_MTL_if @ self.i_stim      # I←Stim
            + self.ext_MTL_i                          # tonic excitability drive    
        )



        # state updates (rectified)
        du_MTL_e_dt = (-self.r_MTL_e + total_inp_MTL_e ) / self.tau_e_MTL
        du_MTL_i_dt = (-self.r_MTL_i + total_inp_MTL_i ) / self.tau_i_MTL
    
        self.u_MTL_e += self.dt * du_MTL_e_dt
        self.u_MTL_i += self.dt * du_MTL_i_dt

        self.r_MTL_e = self.a * self.act(self.u_MTL_e - self.b)
        self.r_MTL_i = self.a * self.act(self.u_MTL_i - self.b)

        if plastic:
            self.W_MTL_MTL_ee += self.dt * (1. / self.tau_ee_MTL) * torch.outer(self.r_MTL_e-self.plast_th, self.r_MTL_e)
            self.W_MTL_MTL_ie += self.dt * (1. / self.tau_ie_MTL) * torch.outer(self.r_MTL_i, self.r_MTL_e )
            self.W_MTL_MTL_ei += self.dt * (1. / self.tau_ei_MTL) * torch.outer(self.r_MTL_e - self.target_e_rate, self.r_MTL_i)
            self.W_MTL_MTL_ii += self.dt * (1. / self.tau_ii_MTL) * torch.outer(self.r_MTL_i - self.target_i_rate, self.r_MTL_i)
            self.W_STIM_MTL_ef += self.dt * (1. / self.tau_ef_MTL) * torch.outer(self.r_MTL_e-self.plast_th, self.e_stim)
            self.W_STIM_MTL_if += self.dt * (1. / self.tau_if_MTL) * torch.outer(self.r_MTL_i, self.i_stim)
            
        self.W_MTL_MTL_ee = torch.clamp(self.W_MTL_MTL_ee, min=0.0,max=.50)
        self.W_MTL_MTL_ie = torch.clamp(self.W_MTL_MTL_ie, min=0.0,max=1.0)
        self.W_MTL_MTL_ei = torch.clamp(self.W_MTL_MTL_ei, min=0.0,max=1.0)
        self.W_MTL_MTL_ii = torch.clamp(self.W_MTL_MTL_ii, min=0.0,max=1.0)
        self.W_STIM_MTL_ef = torch.clamp(self.W_STIM_MTL_ef,min=0.0,max=1.0)
        self.W_STIM_MTL_if = torch.clamp(self.W_STIM_MTL_if,min=0.0,max=1.0)

        return {
            "r_MTL_e": self.r_MTL_e,
            "r_MTL_i": self.r_MTL_i,
        }
    @torch.no_grad()
    def reset_states(self):
        self.u_MTL_e.zero_()
        self.u_MTL_i.zero_()

        # breakpoint()
if __name__ == "__main__":
    device = "cpu"
    # tonic inputs/excitability (shape to match pops; zeros ok)
    nMTL_e, nMTL_i = 10, 10
    ninp = 10
    MTL_inp = torch.zeros(nMTL_e, device=device)    
    ext_MTL_e = torch.abs(torch.normal(0,1,size= (nMTL_e,)))*0#torch.rand(nMTL_e, device=device)
    ext_MTL_i =torch.abs(torch.normal(0,1,size= (nMTL_i,)))*0
    # ext_MTL_e[:100] += 3
    # ext_MTL_i[:25] += 3
    model = TwoAreaEIModel(
        nMTL_e, 
        nMTL_i, 
        ext_MTL_e, 
        ext_MTL_i,
        n_inp=ninp,
        device=device,
        dt = 0.2
    )
    # model.ext_MTL_e[:100] += 2
    # model.ext_MTL_i[:25] += 2

    # Load one MNIST '3' stimulus and set it

    T_total = 100
    stim = get_linearized_mnist_digit3(device=device, train=True, idx=0, binarize=False)
    # model.set_stimulus(stim, gain=1)  # gain controls drive strength
    # model.stim = 
    stim1 = torch.zeros(ninp, device=device)
    stim1[:2] = 5
    stim2 = torch.zeros(ninp, device=device)
    stim2[-2:] = 5
    MTL_weights_ee = [model.W_MTL_MTL_ee.cpu().detach().numpy().copy()]
    MTL_weights_ei = [model.W_MTL_MTL_ei.cpu().detach().numpy().copy()]
    MTL_weights_ie = [model.W_MTL_MTL_ie.cpu().detach().numpy().copy()]
    MTL_weights_ii = [model.W_MTL_MTL_ii.cpu().detach().numpy().copy()]
    STIM_MTL_weights_e = [model.W_STIM_MTL_ef.cpu().detach().numpy().copy()]
    STIM_MTL_weights_i = [model.W_STIM_MTL_if.cpu().detach().numpy().copy()]
    E_rate = []
    I_rate = []
    n_rep = 10
    i1 =0
    for n in range(n_rep):
        for t in trange(T_total):
            if i1%2 ==0:
                model.stim = stim1
            else:
                model.stim = stim2
            i1+=1
            model.i_stim = torch.ones(ninp, device=device)*0.2
            rates = model.step()
            E_rate.append(rates["r_MTL_e"].cpu().detach().numpy().copy())
            I_rate.append(rates["r_MTL_i"].cpu().detach().numpy().copy())
        # MTL_weights_ee.append(model.W_MTL_MTL_ee.cpu().detach().numpy().copy())

    breakpoint()
    MTL_weights_ee.append(model.W_MTL_MTL_ee.cpu().detach().numpy().copy())
    MTL_weights_ei.append(model.W_MTL_MTL_ei.cpu().detach().numpy().copy())
    MTL_weights_ie.append(model.W_MTL_MTL_ie.cpu().detach().numpy().copy())
    MTL_weights_ii.append(model.W_MTL_MTL_ii.cpu().detach().numpy().copy())
    STIM_MTL_weights_e.append(model.W_STIM_MTL_ef.cpu().detach().numpy().copy())
    STIM_MTL_weights_i.append(model.W_STIM_MTL_if.cpu().detach().numpy().copy())
    # # breakpoint()
    T_off = 1000
    # eMTL_e[:1000] -= 3
    # eMTL_i[:250] -= 3
    # eMTL_e[1000:2000] += 3
    # eMTL_i[250:500] += 3
    # stim = get_linearized_mnist_digit3(device=device, train=True, idx=0, binarize=False)
    # model.set_stimulus(stim, gain=2)  # gain controls drive strength
    # model.reset_states()
    # model.ext_MTL_e[:100] -= 2
    # model.ext_MTL_i[:25] -= 2
    # model.ext_MTL_e[100:200] += 2
    # model.ext_MTL_i[25:50] += 2
    for t in trange(T_off):
        if t < 200:
            model.stim = torch.normal(mean = 0,std = 1,size = (ninp,), device=device)
        else:
            model.stim = torch.zeros(ninp, device=device)
        rates = model.step()
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
    # STIM_MTL_weights_e = np.array(STIM_MTL_weights_e)  #   [time, nE, nE]
    # STIM_MTL_weights_i = np.array(STIM_MTL_weights_i)  # [time, nI, nI]

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
    breakpoint()
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

