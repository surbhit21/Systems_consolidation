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
    plt.imshow(x.cpu(), cmap='gray')
    plt.axis('off')
    plt.show()  
    x = x.reshape(-1).to(device)  # [784]
    return x  # [784] in [0,1]

class TwoAreaEIModel:
    def __init__(self, 
                 MTL_inp, CTX_inp,             # constant biases to neurons (vectors)
                 nMTL_e, nCTX_e,               # number of E neurons
                 nMTL_i, nCTX_i,               # number of I neurons
                 eMTL_e, eCTX_e, eMTL_i, eCTX_i,   # external tonic drive/excitability
                 n_inp=784,                    # stimulus dimension (MNIST=28*28)
                 tau_MTL=10.0, tau_CTX=10.0,
                 lr_e_MTL=1e-3, decay_e_MTL=5e-4,
                 lr_e_CTX=1e-4, decay_e_CTX=0.0,
                 lr_i_MTL=1e-3, decay_i_MTL=5e-4,
                 lr_i_CTX=1e-4, decay_i_CTX=0.0,
                 threshold=0.0, dt=1.0, act=torch.relu,
                 device="cpu"
                 ):
        self.device = device

        # Inputs / sizes
        self.MTL_inp = MTL_inp.to(device)  # shape [nMTL_e] or broadcastable
        self.CTX_inp = CTX_inp.to(device)
        self.nMTL_e = nMTL_e
        self.nCTX_e = nCTX_e
        self.nMTL_i = nMTL_i
        self.nCTX_i = nCTX_i
        self.n_inp  = n_inp

        # Dynamics
        self.tau_MTL = tau_MTL
        self.tau_CTX = tau_CTX
        self.dt = dt
        self.act = act
        self.threshold = threshold

        # Learning rates/decay
        self.lr_e_MTL = lr_e_MTL
        self.decay_e_MTL = decay_e_MTL
        self.lr_e_CTX = lr_e_CTX
        self.decay_e_CTX = decay_e_CTX
        self.lr_i_MTL = lr_i_MTL
        self.decay_i_MTL = decay_i_MTL
        self.lr_i_CTX = lr_i_CTX
        self.decay_i_CTX = decay_i_CTX

        # External tonic drives
        self.ext_MTL_e = eMTL_e.to(device)
        self.ext_CTX_e = eCTX_e.to(device)
        self.ext_MTL_i = eMTL_i.to(device)
        self.ext_CTX_i = eCTX_i.to(device)

        # States
        self.r_MTL_e = torch.zeros(nMTL_e, device=device)
        self.r_CTX_e = torch.zeros(nCTX_e, device=device)
        self.r_MTL_i = torch.zeros(nMTL_i, device=device)
        self.r_CTX_i = torch.zeros(nCTX_i, device=device)

        # Target rates for inhibitory plasticity
        self.rho_MTL_e = torch.ones(nMTL_e, device=device)
        self.rho_CTX_e = torch.ones(nCTX_e, device=device)
        # Recurrent weights (E/I blocks) — shapes corrected
        # MTL
        self.W_MTL_MTL_ee = torch.zeros(nMTL_e, nMTL_e, device=device)   # E←E
        self.W_MTL_MTL_ei = torch.zeros(nMTL_e, nMTL_i, device=device)   # E←I
        self.W_MTL_MTL_ie = torch.zeros(nMTL_i, nMTL_e, device=device)   # I←E
        self.W_MTL_MTL_ii = -torch.abs(torch.randn(nMTL_i, nMTL_i, device=device)) / torch.sqrt(torch.tensor(nMTL_i, dtype=torch.float, device=device))  # I←I

        # CTX
        self.W_CTX_CTX_ee = torch.zeros(nCTX_e, nCTX_e, device=device)   # E←E
        self.W_CTX_CTX_ei = torch.zeros(nCTX_e, nCTX_i, device=device)   # E←I
        self.W_CTX_CTX_ie = torch.zeros(nCTX_i, nCTX_e, device=device)   # I←E
        self.W_CTX_CTX_ii = -torch.abs(torch.randn(nCTX_i, nCTX_i, device=device)) / torch.sqrt(torch.tensor(nCTX_i, dtype=torch.float, device=device))  # I←I

        # Stimulus feedforward weights (stim -> E)
        # Small random initial fan-in
        scale_MTL = 1.0 / (n_inp ** 0.5)
        scale_CTX = 1.0 / (n_inp ** 0.5)
        self.W_STIM_MTL_e = 0.1 * torch.randn(nMTL_e, n_inp, device=device) * scale_MTL
        self.W_STIM_CTX_e = 0.1 * torch.randn(nCTX_e, n_inp, device=device) * scale_CTX

        # Current stimulus vector [n_inp]; default zeros
        self.stim = torch.zeros(n_inp, device=device)

    @torch.no_grad()
    def set_stimulus(self, stim_vec, normalize=True, clip=True, gain=1.0):
        """
        stim_vec: tensor-like [n_inp]. Typically a flattened MNIST image.
        """
        s = torch.as_tensor(stim_vec, device=self.device, dtype=torch.float32).reshape(-1)
        if s.numel() != self.n_inp:
            raise ValueError(f"stim size {s.numel()} != n_inp {self.n_inp}")
        if clip:
            s = torch.clamp(s, 0.0, 1.0)
        if normalize:
            # optional norm scaling so different images drive similarly
            denom = (s.norm(p=2) + 1e-8)
            s = s / denom
        self.stim = gain * s
        x = self.stim.reshape(28, 28)
        print(x)
        plt.imshow(x.cpu(), cmap='gray')
        plt.axis('off')
        plt.show() 

    def step(self):
        """
        One Euler step of rate dynamics.
        """
        # Inputs to MTL
        total_inp_MTL_e = (
            self.W_MTL_MTL_ee @ self.r_MTL_e          # E←E
            - self.W_MTL_MTL_ei @ self.r_MTL_i        # E←I
            + self.W_STIM_MTL_e @ self.stim           # E←Stim
            + self.ext_MTL_e
        )
        total_inp_MTL_i = (
            self.W_MTL_MTL_ie @ self.r_MTL_e          # I←E
            - self.W_MTL_MTL_ii @ self.r_MTL_i        # I←I (note W_ii is negative)
            + self.ext_MTL_i   # allow broadcasting/slicing
        )

        # Inputs to CTX
        total_inp_CTX_e = (
            self.W_CTX_CTX_ee @ self.r_CTX_e
            - self.W_CTX_CTX_ei @ self.r_CTX_i
            + self.W_STIM_CTX_e @ self.stim
            + self.ext_CTX_e 
        )
        total_inp_CTX_i = (
            self.W_CTX_CTX_ie @ self.r_CTX_e
            - self.W_CTX_CTX_ii @ self.r_CTX_i
            + self.ext_CTX_i
        )

        # Rate updates (rectified)
        dr_MTL_e_dt = (-self.r_MTL_e + self.act(total_inp_MTL_e - self.threshold)) / self.tau_MTL
        dr_MTL_i_dt = (-self.r_MTL_i + self.act(total_inp_MTL_i - self.threshold)) / self.tau_MTL
        dr_CTX_e_dt = (-self.r_CTX_e + self.act(total_inp_CTX_e - self.threshold)) / self.tau_CTX
        dr_CTX_i_dt = (-self.r_CTX_i + self.act(total_inp_CTX_i - self.threshold)) / self.tau_CTX

        self.r_MTL_e += self.dt * dr_MTL_e_dt
        self.r_MTL_i += self.dt * dr_MTL_i_dt
        self.r_CTX_e += self.dt * dr_CTX_e_dt
        self.r_CTX_i += self.dt * dr_CTX_i_dt

        # --- Rate updates ---
        dr_MTL_e_dt = (-self.r_MTL_e + self.act(total_inp_MTL_e - self.threshold)) / self.tau_MTL
        dr_MTL_i_dt = (-self.r_MTL_i + self.act(total_inp_MTL_i - self.threshold)) / self.tau_MTL
        dr_CTX_e_dt = (-self.r_CTX_e + self.act(total_inp_CTX_e - self.threshold)) / self.tau_CTX
        dr_CTX_i_dt = (-self.r_CTX_i + self.act(total_inp_CTX_i - self.threshold)) / self.tau_CTX

        self.r_MTL_e += self.dt * dr_MTL_e_dt
        self.r_MTL_i += self.dt * dr_MTL_i_dt
        self.r_CTX_e += self.dt * dr_CTX_e_dt
        self.r_CTX_i += self.dt * dr_CTX_i_dt

        # --- Plasticity ---
        # 1) EE Hebbian + decay (optional, keep if you want)
        self.W_MTL_MTL_ee += self.dt * (self.lr_e_MTL * torch.outer(self.r_MTL_e, self.r_MTL_e) - self.decay_e_MTL * self.W_MTL_MTL_ee)
        self.W_CTX_CTX_ee += self.dt * (self.lr_e_CTX * torch.outer(self.r_CTX_e, self.r_CTX_e) - self.decay_e_CTX * self.W_CTX_CTX_ee)

        # 2) II STATIC  -> NO UPDATE to W_*_ii

        # 3) Vogels inhibitory plasticity on I→E (E←I blocks)
        # ΔW_ei = η * (rE - ρ)[:,None] * (rI)[None,:]
        # breakpoint()
        dW_MTL_ei = self.lr_i_MTL * torch.outer(self.r_MTL_e - self.rho_MTL_e, self.r_MTL_i)
        dW_CTX_ei = self.lr_i_CTX * torch.outer(self.r_CTX_e - self.rho_CTX_e, self.r_CTX_i)

        self.W_MTL_MTL_ei += self.dt * dW_MTL_ei
        self.W_CTX_CTX_ei += self.dt * dW_CTX_ei

        # 3) Excitatory plasticity on E→I 
        dW_MTL_ie = self.lr_e_MTL * torch.outer(self.r_MTL_i, self.r_MTL_e) - self.decay_i_MTL * self.W_MTL_MTL_ie
        dW_CTX_ie = self.lr_e_CTX * torch.outer(self.r_CTX_i, self.r_CTX_e) - self.decay_i_CTX * self.W_CTX_CTX_ie

        self.W_MTL_MTL_ie += self.dt * dW_MTL_ie
        self.W_CTX_CTX_ie += self.dt * dW_CTX_ie


        # enforce nonnegativity and upper bound (inhibition magnitude)
        # self.W_MTL_MTL_ei.clamp_(min=0.0, max=self.w_ie_max)
        # self.W_CTX_CTX_ei.clamp_(min=0.0, max=-1)


        return {
            "r_MTL_e": self.r_MTL_e,
            "r_MTL_i": self.r_MTL_i,
            "r_CTX_e": self.r_CTX_e,
            "r_CTX_i": self.r_CTX_i,
        }

# ---------------------------
# Minimal usage example
# ---------------------------
if __name__ == "__main__":
    device = "cpu"
    # tonic inputs/excitability (shape to match pops; zeros ok)
    nMTL_e, nMTL_i = 728, 182
    nCTX_e, nCTX_i = 728, 182
    MTL_inp = torch.zeros(nMTL_e, device=device)
    CTX_inp = torch.zeros(nCTX_e, device=device)
    eMTL_e = torch.zeros(nMTL_e, device=device)
    eCTX_e = torch.zeros(nCTX_e, device=device)
    eMTL_i = torch.zeros(nMTL_i, device=device)
    eCTX_i = torch.zeros(nCTX_i, device=device)

    model = TwoAreaEIModel(
        MTL_inp, CTX_inp,
        nMTL_e, nCTX_e,
        nMTL_i, nCTX_i,
        eMTL_e, eCTX_e, eMTL_i, eCTX_i,
        n_inp=784,
        device=device
    )

    # Load one MNIST '3' stimulus and set it
    stim = get_linearized_mnist_digit3(device=device, train=True, idx=0, binarize=False)
    model.set_stimulus(stim, normalize=True, gain=50.0)  # gain controls drive strength

    # Run a few steps
    for t in range(60):
        out = model.step()
    breakpoint()
    print("MTL E mean rate:", model.r_MTL_e.mean().item())
    print("CTX E mean rate:", model.r_CTX_e.mean().item())


