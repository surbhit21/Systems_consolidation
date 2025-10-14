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


def get_grating_input(N, c, Af, sigma_f, stimulus_angle, pref_orientations):
    
    assert  N == pref_orientations.shape[0]
    # print(pref_orientations)
    # Compute circular difference (shortest angle distance)
    def circ_diff(a, b):
        d = torch.abs(a - b)
        return torch.minimum(d, 180 - d)

    # Firing rates
    delta = circ_diff(pref_orientations, stimulus_angle)
    rates =  c*Af * torch.exp(-0.5 * (delta / sigma_f)**2)
    rates = torch.clip(rates, 0, None)  # no negative rates
    return rates

class TwoAreaEIModel:
    def __init__(self, 
                 nMTL_e,               # number of E neurons
                 nMTL_i,               # number of I neurons
                 eMTL_e, eMTL_i,       # tonic excitability drives (shape [nMTL_e], [nMTL_i])
                 n_inp = 784,                    # stimulus dimension (MNIST=28*28)
                 tau_e_MTL = 2,
                 tau_i_MTL = 1.7,
                 tau_ee_MTL = 1/2e-6,
                 tau_ei_MTL = 1/4e-6,
                 tau_ii_MTL = 1/5e-6, 
                 tau_ie_MTL = 1/3e-6,
                 tau_ef_MTL = 1/2e-6,
                 tau_if_MTL = 1/4e-6,
                #  lr_ef_MTL = 4e-5,
                #  lr_if_MTL = 8e-5,
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
        self.a = 0.04  # scaling for inhibitory plasticity
        self.b = 0
        self.n = 2

        # Learning rates/decay
        self.tau_ee_MTL = tau_ee_MTL
        self.tau_ei_MTL = tau_ei_MTL
        self.tau_ii_MTL = tau_ii_MTL
        self.tau_ie_MTL = tau_ie_MTL
        self.tau_ef_MTL = tau_ef_MTL
        self.tau_if_MTL = tau_if_MTL
        # self.lr_ef_MTL = lr_ef_MTL

        # self.lr_if_MTL = lr_if_MTL

        # External tonic drives
        self.ext_MTL_e = eMTL_e.to(device)
        self.ext_MTL_i = eMTL_i.to(device)

        # states
        self.u_MTL_e = torch.zeros(nMTL_e, device=device)  # membrane potentials
        self.u_MTL_i = torch.zeros(nMTL_i, device=device)

        # rates = a*[u-b]_+^ 2
        self.r_MTL_e = torch.zeros(nMTL_e, device=device)
        self.r_MTL_i = torch.zeros(nMTL_i, device=device)

        # Target rates for inhibitory plasticity
        self.rho_MTL_e = 2*torch.ones(nMTL_e, device=device)
        
        # Recurrent weights (E/I blocks) — shapes corrected
        # MTL
        mu, std = 0.1,0.05
        self.W_MTL_MTL_ee = torch.zeros(nMTL_e,nMTL_e)#torch.abs(torch.normal(mu,std,size=(nMTL_e, nMTL_e)))   # E ← E
        self.W_MTL_MTL_ei = torch.abs(torch.normal(mu,std,size=(nMTL_e, nMTL_i)))   # E ← I
        self.W_MTL_MTL_ie = torch.abs(torch.normal(mu,std,size=(nMTL_i, nMTL_e)))   # I ← E
        self.W_MTL_MTL_ii = torch.abs(torch.normal(mu,std,size=(nMTL_i, nMTL_i))) #torch.zeros(torch.randn(nMTL_i, nMTL_i, device=device)) / torch.sqrt(torch.tensor(nMTL_i, dtype=torch.float, device=device))  # I←I

        # Target sums for weight normalization 
        self.W_EE = 1  # target sum of EE weights onto each E neuron
        self.W_EI = 0.4 # target sum of EI weights onto each E neuron
        self.W_IE = 2 # target sum of IE weights onto eacch E neuron
        self.W_II = 0.65 # target sum of IE weights onto each E neuron

        # Stimulus feedforward weights (stim -> E)
        # Small random initial fan-in
        # scale_MTL = 1.0 / (n_inp ** 0.5)
        self.W_STIM_MTL_ef = torch.abs(torch.normal(0.1,0.05,size=(nMTL_e,n_inp))) 
        self.W_STIM_MTL_if = torch.abs(torch.normal(0.1,0.05,size=(nMTL_i,n_inp))) 
        self.normalize_weights()
        # self.W_STIM_MTL_ef = self.W_STIM_MTL_ef/ (self.W_STIM_MTL_ef.sum(dim=1, keepdim=True)) 
        # self.W_STIM_MTL_if = self.W_STIM_MTL_if/ (self.W_STIM_MTL_if.sum(dim=1, keepdim=True))   
        # breakpoint()
        # self.W_EF = None  # target sum of feedforward weights onto each E neuron
        # self.W_IF = None  # target sum of feedforward weights onto each I neuron

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
            + self.W_STIM_MTL_ef @ self.stim            # E←Stim
            + self.ext_MTL_e                          # tonic excitability drive  
        )
        total_inp_MTL_i = (
            self.W_MTL_MTL_ie @ self.r_MTL_e          # I←E
            - self.W_MTL_MTL_ii @ self.r_MTL_i        # I←I (note W_ii is positive, so minus sign here)
            + self.W_STIM_MTL_if @ self.stim      # I←Stim
            + self.ext_MTL_i                          # tonic excitability drive    
        )


        # print(self.u_MTL_e.mean(),self.u_MTL_i.mean())
        # state updates (rectified)
        du_MTL_e_dt = (-self.u_MTL_e + total_inp_MTL_e ) / self.tau_e_MTL
        du_MTL_i_dt = (-self.u_MTL_i + total_inp_MTL_i ) / self.tau_i_MTL
    
        self.u_MTL_e += self.dt * du_MTL_e_dt
        self.u_MTL_i += self.dt * du_MTL_i_dt

        self.r_MTL_e = self.a * torch.pow(self.act(self.u_MTL_e - self.b), self.n)
        self.r_MTL_i = self.a * torch.pow(self.act(self.u_MTL_i - self.b), self.n)

        if self.r_MTL_e.max() > 1000 or self.r_MTL_i.max() > 1000:
            print("too high rates!")
            breakpoint()
        if plastic:
            self.W_MTL_MTL_ee += self.dt * (1. / self.tau_ee_MTL) * torch.outer(self.r_MTL_e, self.r_MTL_e)
            self.W_MTL_MTL_ie += self.dt * (1. / self.tau_ie_MTL) * torch.outer(self.r_MTL_i, self.r_MTL_e )
            self.W_MTL_MTL_ei += self.dt * (1. / self.tau_ei_MTL) * torch.outer(self.r_MTL_e, self.r_MTL_i)
            self.W_MTL_MTL_ii += self.dt * (1. / self.tau_ii_MTL) * torch.outer(self.r_MTL_i, self.r_MTL_i)
            self.W_STIM_MTL_ef += self.dt * (1. / self.tau_ef_MTL) * torch.outer(self.r_MTL_e, self.stim)
            self.W_STIM_MTL_if += self.dt * (1. / self.tau_if_MTL) * torch.outer(self.r_MTL_i, self.stim)
            if normalize:
                self.normalize_weights()
        # self.W_MTL_MTL_ee = torch.clamp(self.W_MTL_MTL_ee, min=0.0,max=1.0)
        # self.W_MTL_MTL_ie = torch.clamp(self.W_MTL_MTL_ie, min=0.0,max=1.0)
        # self.W_MTL_MTL_ei = torch.clamp(self.W_MTL_MTL_ei, min=0.0,max=1.0)
        # self.W_MTL_MTL_ii = torch.clamp(self.W_MTL_MTL_ii, min=0.0,max=1.0)
        # self.W_STIM_MTL_ef = torch.clamp(self.W_STIM_MTL_ef,min=0.0,max=1.0)
        # self.W_STIM_MTL_if = torch.clamp(self.W_STIM_MTL_if,min=0.0,max=1.0)
        return {
            "r_MTL_e": self.r_MTL_e,
            "r_MTL_i": self.r_MTL_i,
        }
    @torch.no_grad()
    def reset_states(self):
        self.u_MTL_e.zero_()
        self.u_MTL_i.zero_()

    @torch.no_grad()
    def feedforward_sums(self):
        """
        Returns the sum of all feedforward (Stim→*) weights
        onto each postsynaptic neuron.
        E_sums: shape [nMTL_e]
        I_sums: shape [nMTL_i]
        """
        EF_sums = self.W_STIM_MTL_ef.sum(dim=1,keepdim=True)          # per E neuron
        IF_sums = self.W_STIM_MTL_if.sum(dim=1,keepdim=True)          # per I neuron
        return {"E": EF_sums, "I": IF_sums}
    @torch.no_grad()
    def normalize_weights(self ):
        """
        Normalize incoming weights to each neuron to fixed sum if the sum is defined (>0).
        
        This is done for all 6 weight matrices.
        """
        eps = 1e-12  # to avoid div0
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
            scale_factors_e = (self.W_EE / (total_weights_e + eps))
            self.W_MTL_MTL_ee *= scale_factors_e
            self.W_STIM_MTL_ef *= scale_factors_e
        if self.W_IE is not None:
            scale_fcators_i = (self.W_IE / (total_weights_i + eps)) 
            self.W_MTL_MTL_ie *=  scale_fcators_i
            self.W_STIM_MTL_if *= scale_fcators_i
        if self.W_EI is not None:
            self.W_MTL_MTL_ei *= (self.W_EI / (total_EI+eps))
        if self.W_II is not None:
            self.W_MTL_MTL_ii *= (self.W_II / (total_II+eps))
        
        # self.W_MTL_MTL_ee.clamp_(min=0); self.W_STIM_MTL_ef.clamp_(min=0)
        # self.W_MTL_MTL_ie.clamp_(min=0); self.W_STIM_MTL_if.clamp_(min=0)
        # self.W_MTL_MTL_ei.clamp_(min=0); self.W_MTL_MTL_ii.clamp_(min=0)


    

        # breakpoint()
if __name__ == "__main__":
    device = "cpu"
    # tonic inputs/excitability (shape to match pops; zeros ok)
    nMTL_e, nMTL_i = 20,20
    ninp = 20
    MTL_inp = torch.zeros(nMTL_e, device=device)    
    ext_MTL_e = torch.abs(torch.normal(0,.1,size= (nMTL_e,)))#torch.rand(nMTL_e, device=device)
    ext_MTL_i =torch.abs(torch.normal(0,.1,size= (nMTL_i,)))

    c = 1
    Af = 35
    sigma_f = 12
    # stimulus_angle = 30
    # Preferred orientations of neurons
    pref_orientations = torch.linspace(0, 180, ninp, device=device)

    # ext_MTL_e[:20] += 3
    # ext_MTL_i[:5] += 3
    model = TwoAreaEIModel(
        nMTL_e, 
        nMTL_i, 
        ext_MTL_e, 
        ext_MTL_i,
        n_inp=ninp,
        device=device,
        dt = 1
    )
    # model.ext_MTL_e[:10] += 5
    # model.ext_MTL_i[:10] += 5

    # Load one MNIST '3' stimulus and set it

    T_1_stim = 100
    # model.stim[:5] += 3

    MTL_weights_ee = [model.W_MTL_MTL_ee.cpu().detach().numpy().copy()]
    MTL_weights_ei = [model.W_MTL_MTL_ei.cpu().detach().numpy().copy()]
    MTL_weights_ie = [model.W_MTL_MTL_ie.cpu().detach().numpy().copy()]
    MTL_weights_ii = [model.W_MTL_MTL_ii.cpu().detach().numpy().copy()]
    STIM_MTL_weights_e = [model.W_STIM_MTL_ef.cpu().detach().numpy().copy()]
    STIM_MTL_weights_i = [model.W_STIM_MTL_if.cpu().detach().numpy().copy()]
    E_rate = []
    I_rate = []
    ext_MTL_e = []
    ext_MTL_i = []
    Nstim = 10**5
    # WF_sums = [model.feedforward_sums()]
    input_samples = np.random.uniform(0,180,size = Nstim)
    # print(input_samples)
    for i in trange(Nstim):
        Inp_rates = get_grating_input(ninp, c, Af, sigma_f, input_samples[i], pref_orientations)
        model.stim = Inp_rates
        for t in range(T_1_stim):
            # model.stim = MTL_inp
            rates = model.step()
            E_rate.append(rates["r_MTL_e"].cpu().detach().numpy().copy())
            I_rate.append(rates["r_MTL_i"].cpu().detach().numpy().copy())
            ext_MTL_e.append(model.ext_MTL_e.cpu().detach().numpy().copy())
            ext_MTL_i.append(model.ext_MTL_i.cpu().detach().numpy().copy())
            # MTL_weights_ee.append(model.W_MTL_MTL_ee.cpu().detach().numpy().copy())

    # WF_sums.append(model.feedforward_sums())
    # breakpoint()

    MTL_weights_ee.append(model.W_MTL_MTL_ee.cpu().detach().numpy().copy())
    MTL_weights_ei.append(model.W_MTL_MTL_ei.cpu().detach().numpy().copy())
    MTL_weights_ie.append(model.W_MTL_MTL_ie.cpu().detach().numpy().copy())
    MTL_weights_ii.append(model.W_MTL_MTL_ii.cpu().detach().numpy().copy())
    STIM_MTL_weights_e.append(model.W_STIM_MTL_ef.cpu().detach().numpy().copy())
    STIM_MTL_weights_i.append(model.W_STIM_MTL_if.cpu().detach().numpy().copy())


    # for k in ["E","I"]:
    # # breakpoint()
    # T_off = 1000
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
    # for t in trange(T_off):
    #     if t < 200:
    #         model.stim = torch.normal(mean = 0,std = 1,size = (ninp,), device=device)
    #     else:
    #         model.stim = torch.zeros(ninp, device=device)
    #     rates = model.step(plastic=False,normalize=False)
    #     E_rate.append(rates["r_MTL_e"].cpu().detach().numpy().copy())
    #     I_rate.append(rates["r_MTL_i"].cpu().detach().numpy().copy())
    
    
    # MTL_weights_ee.append(model.W_MTL_MTL_ee.cpu().detach().numpy().copy())
    # MTL_weights_ei.append(model.W_MTL_MTL_ei.cpu().detach().numpy().copy())
    # MTL_weights_ie.append(model.W_MTL_MTL_ie.cpu().detach().numpy().copy())
    # MTL_weights_ii.append(model.W_MTL_MTL_ii.cpu().detach().numpy().copy())
    # STIM_MTL_weights_e.append(model.W_STIM_MTL_ef.cpu().detach().numpy().copy())
    # STIM_MTL_weights_i.append(model.W_STIM_MTL_if.cpu().detach().numpy().copy())
    



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
    op_folder_data = "./data/HPC_W_Norm_noext"
    op_folder_plots = "./plots/HPC_W_Norm_noext"
    os.makedirs(op_folder_data, exist_ok=True)
    os.makedirs(op_folder_plots, exist_ok=True)
    np.savez("{}/EI_HPC_with_normalization.npz".format(op_folder_data),
            MTL_weights_ee=MTL_weights_ee,
            MTL_weights_ei=MTL_weights_ei,
            MTL_weights_ie=MTL_weights_ie,
            MTL_weights_ii=MTL_inp_weights_ii,
            STIM_MTL_weights_e=STIM_MTL_weights_e,
            STIM_MTL_weights_i=STIM_MTL_weights_i,
            E_rate=E_rate,
            I_rate=I_rate,
            ext_MTL_i=ext_MTL_i,
            ext_MTL_e=ext_MTL_e
            )
    
    # 
