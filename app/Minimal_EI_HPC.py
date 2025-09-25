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
        self.W_MTL_MTL_ee = torch.zeros(nMTL_e, nMTL_e, device=device)   # E←E
        self.W_MTL_MTL_ei = torch.zeros(nMTL_e, nMTL_i, device=device)   # E←I
        self.W_MTL_MTL_ie = torch.zeros(nMTL_i, nMTL_e, device=device)   # I←E
        self.W_MTL_MTL_ii = -torch.abs(torch.randn(nMTL_i, nMTL_i, device=device)) / torch.sqrt(torch.tensor(nMTL_i, dtype=torch.float, device=device))  # I←I


        # Stimulus feedforward weights (stim -> E)
        # Small random initial fan-in
        scale_MTL = 1.0 / (n_inp ** 0.5)
        self.W_STIM_MTL_e = 0.1 * torch.randn(nMTL_e, n_inp, device=device) * scale_MTL

        self.target_ee_sum = float(1)  # target sum of EE weights onto each E neuron
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
        # print(x)
        # plt.imshow(x.cpu(), cmap='gray')
        # plt.axis('off')
        # plt.show() 

    def step(self,plastic=True):
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

        # Rate updates (rectified)
        dr_MTL_e_dt = (-self.r_MTL_e + self.act(total_inp_MTL_e - self.threshold)) / self.tau_MTL
        dr_MTL_i_dt = (-self.r_MTL_i + self.act(total_inp_MTL_i - self.threshold)) / self.tau_MTL
    
        
        self.r_MTL_e += self.dt * dr_MTL_e_dt
        self.r_MTL_i += self.dt * dr_MTL_i_dt

        if plastic:
            # --- Plasticity ---
            # 1) EE Hebbian + decay (optional, keep if you want)
            self.W_MTL_MTL_ee += self.dt * (self.lr_e_MTL * torch.outer(self.r_MTL_e, self.r_MTL_e) - self.decay_e_MTL * self.W_MTL_MTL_ee)
            self._normalize_ee_rows()  # keep E←E weights normalized
            # 2) II STATIC  -> NO UPDATE to W_*_ii

            # 3) Vogels inhibitory plasticity on I→E (E←I blocks)
            # ΔW_ei = η * (rE - ρ)[:,None] * (rI)[None,:]
            # breakpoint()
            dW_MTL_ei = self.lr_i_MTL * torch.outer(self.r_MTL_e - self.rho_MTL_e, self.r_MTL_i)
        
            self.W_MTL_MTL_ei += self.dt * dW_MTL_ei
            
            # 3) Excitatory plasticity on E→I 
            dW_MTL_ie = self.lr_e_MTL * torch.outer(self.r_MTL_i, self.r_MTL_e) - self.decay_i_MTL * self.W_MTL_MTL_ie
            
            self.W_MTL_MTL_ie += self.dt * dW_MTL_ie
            
            # self._normalize_ee_rows(conn="IE")  # keep E→I weights normalized


            # enforce nonnegativity and upper bound (inhibition magnitude)
            # self.W_MTL_MTL_ei.clamp_(min=0.0, max=self.w_ie_max)
            # self.W_CTX_CTX_ei.clamp_(min=0.0, max=-1)


        return {
            "r_MTL_e": self.r_MTL_e,
            "r_MTL_i": self.r_MTL_i,
        }

    @torch.no_grad()
    def _normalize_ee_rows(self, conn = "EE", eps: float = 1e-8):
        """
        Row-wise L1 renormalization of E←E weights so that
        each postsynaptic E neuron has sum_j W_ee[i,j] == target_ee_sum.
        Rows = postsyn E, columns = presyn E (E←E).
        """
        if conn == "EE":
            W = self.W_MTL_MTL_ee
        else:
            W = self.W_MTL_MTL_ie

        # Keep excitatory nonnegative
        W.clamp_(min=0.0)

        # Current row sums (incoming E mass per E neuron)
        sums = W.sum(dim=1, keepdim=True)

        # If a row is all zeros, initialize it uniformly to the target
        zero_mask = (sums <= eps).squeeze(1)
        if zero_mask.any():
            W[zero_mask, :] = self.target_ee_sum / self.nMTL_e
            sums = W.sum(dim=1, keepdim=True)  # recompute

        # Exact multiplicative renormalization to the target
        scale = self.target_ee_sum / (sums + eps)
        W *= scale

@torch.no_grad()
def make_partial(x: torch.Tensor, keep_frac: float) -> torch.Tensor:
    mask = (torch.rand_like(x) < keep_frac).float()
    return x * mask

@torch.no_grad()
def make_noisy(x: torch.Tensor, std: float = 0.3) -> torch.Tensor:
    return (x + std * torch.randn_like(x)).clamp(0.0, 1.0)

@torch.no_grad()
def fit_readout_ridge(R: torch.Tensor, Y: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    """
    R: [T, nE]   (E rates over time)
    Y: [T, 784]  (target pixels over time)
    Returns Wout: [784, nE] such that y_hat = Wout @ r_E
    """
    Rt = R.t()                        # [nE, T]
    G = Rt @ R                        # [nE, nE]
    A = Y.t() @ R                     # [784, nE]
    I = torch.eye(G.shape[0], device=R.device)
    Wout = A @ torch.linalg.solve(G + lam * I, I)
    return Wout

@torch.no_grad()
def run_recall(model, Wout, cue, steps=400, cue_gain=20.0, normalize_cue=True):
    model.set_stimulus(cue, normalize=normalize_cue, gain=cue_gain)
    r_T, y_T = [], []
    for _ in range(steps):
        model.step(plastic=False)
        r = model.r_MTL_e.detach().clone()
        y = (Wout @ r).clamp(0.0, 1.0)
        r_T.append(r); y_T.append(y)
    return torch.stack(r_T, 0), torch.stack(y_T, 0)  # [T,nE], [T,784]

def plot_pattern_completion(model, x_target, Wout, cues, steps=400, cue_gain=20.0):
    assert isinstance(cues, (list, tuple)) and len(cues) > 0, "cues is empty"

    rows = len(cues)
    fig, axes = plt.subplots(rows, 4, figsize=(11, 2.2*rows))
    # make axes always 2D for consistent indexing
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for row, (name, cue) in enumerate(cues):
        # run recall
        r_T, y_T = run_recall(model, Wout, cue, steps=steps, cue_gain=cue_gain, normalize_cue=True)
        assert y_T.ndim == 2 and y_T.shape[1] == 28*28, f"bad y_T shape {y_T.shape}"

        # 1) cue
        axes[row, 0].imshow(cue.view(28,28).cpu(), cmap="gray")
        axes[row, 0].set_title(f"Cue: {name}"); axes[row, 0].axis("off")

        # 2) decoded at end
        axes[row, 1].imshow(y_T[-1].view(28,28).cpu(), cmap="gray")
        axes[row, 1].set_title("Decoded @ end"); axes[row, 1].axis("off")

        # 3) cosine to target
        target_rep = x_target.unsqueeze(0).expand_as(y_T)
        cos = F.cosine_similarity(y_T, target_rep, dim=1).cpu().numpy()
        axes[row, 2].plot(cos); axes[row, 2].set_title("cosine(y, target)")
        axes[row, 2].set_xlabel("time")

        # 4) target
        axes[row, 3].imshow(x_target.view(28,28).cpu(), cmap="gray")
        axes[row, 3].set_title("Target 3"); axes[row, 3].axis("off")

    fig.tight_layout(); plt.show()

# ---------------------------
# Minimal usage example
# ---------------------------
if __name__ == "__main__":
    device = "cpu"
    # tonic inputs/excitability (shape to match pops; zeros ok)
    nMTL_e, nMTL_i = 4000, 1000
    MTL_inp = torch.zeros(nMTL_e, device=device)
    eMTL_e = torch.zeros(nMTL_e, device=device)
    eMTL_i = torch.zeros(nMTL_i, device=device)
    
    model = TwoAreaEIModel(
        nMTL_e, 
        nMTL_i, 
        n_inp=784,
        device=device
    )

    # Load one MNIST '3' stimulus and set it
    stim = get_linearized_mnist_digit3(device=device, train=True, idx=0, binarize=False)
    model.set_stimulus(stim, normalize=True, gain=200.0)  # gain controls drive strength

    T_total = 2000
    K_tail  = 400                         # last steps to use for readout fitting
    R_list, Y_list = [], []

    MTL_weights_ee = [model.W_MTL_MTL_ee.cpu().detach().numpy().copy()]
    MTL_weights_ei = [model.W_MTL_MTL_ei.cpu().detach().numpy().copy()]
    MTL_weights_ie = [model.W_MTL_MTL_ie.cpu().detach().numpy().copy()]
    E_rate = []
    I_rate = []
    for t in trange(T_total):
        rates = model.step()
        E_rate.append(rates["r_MTL_e"].cpu().numpy())
        I_rate.append(rates["r_MTL_i"].cpu().numpy())
        if t >= T_total - K_tail:
            R_list.append(model.r_MTL_e.detach().clone().unsqueeze(0))  # [1, nE]
            Y_list.append(stim.view(1, -1))                          # [1, 784]

    MTL_weights_ee.append(model.W_MTL_MTL_ee.cpu().detach().numpy().copy())
    MTL_weights_ei.append(model.W_MTL_MTL_ei.cpu().detach().numpy().copy())
    MTL_weights_ie.append(model.W_MTL_MTL_ie.cpu().detach().numpy().copy())
   
    MTL_weights_ei = np.array(MTL_weights_ei)  # [time, nE, nI]
    MTL_weights_ie = np.array(MTL_weights_ie)  # [time, nI, nE]
    MTL_weights_ee = np.array(MTL_weights_ee)  # [
    
    E_rate = np.array(E_rate)  # [time, nE]
    I_rate = np.array(I_rate)  # [time, nI]
    R_train = torch.cat(R_list, dim=0)  # [K, nE]
    Y_train = torch.cat(Y_list, dim=0)  # [K, 784]
    # --- Fit linear readout (E rates -> pixels) ---
    Wout = fit_readout_ridge(R_train, Y_train, lam=1e-2)  # [784, nE]

    # Check if the Wouts are learned  correctly
    Yhat1 = torch.rand_like(model.r_MTL_e.detach().clone().unsqueeze(0)) @ Wout.T # [K, 784]

    # K = 200
    # R_train = torch.stack([model.r_MTL_e.detach().clone() for _ in range(K)], 0)  # or your buffer
    # Y_train = stim.view(1,-1).expand(K, -1)
    # Wout = fit_readout_ridge(R_train, Y_train, lam=1e-2)
    # breakpoint()
    plt.figure(figsize=(6,6))
    plt.imshow(Yhat1.reshape(28,28), cmap='gray')
    plt.show() 

    
    cues = [
        ("partial_50", make_partial(stim, 0.5)),
        ("partial_30", make_partial(stim, 0.3)),
        ("partial_10", make_partial(stim, 0.1)),
        ("noisy",      make_noisy(stim, 0.3)),
        ("total_noise", torch.rand_like(stim)),
    ]
    print("num cues:", len(cues))

    recall_with_or_without_recurrence(model, Wout, stim, stim, steps=400, cue_gain=200.0, kill_recurrence=True)
    plot_pattern_completion(model, stim, Wout, cues, steps=400, cue_gain=200.0)
   
#     # Run a few steps
#     for t in trange(2000):
#         rates = model.step()
#         E_rate.append(rates["r_MTL_e"].cpu().numpy())
#         I_rate.append(rates["r_MTL_i"].cpu().numpy())

# breakpoint()
plot_activity_n_excitability_time([E_rate.T,I_rate.T],
                    titles=['Neuronal Activity (Excitatory)','Neuronal Activity (Inhibitory)'],
                    fname="./plots/HPC_RNN_MNIST/Activity.png",
                    cmaps=['Greens','Blues'])
labs = ["Initial weights","Final weights"]
plot_weights_over_time(MTL_weights_ee,
                       titles= labs,
                       fname="./plots/HPC_RNN_MNIST/Rec_w_MT_ee.png",
                       cmaps='gray_r')
plot_weights_over_time(MTL_weights_ei,
                       titles= labs,
                       fname="./plots/HPC_RNN_MNIST/Rec_w_MT_ei.png",
                       cmaps='gray_r')
plot_weights_over_time(MTL_weights_ie,
                       titles= labs,
                       fname="./plots/HPC_RNN_MNIST/Rec_w_MT_ie.png",
                       cmaps='gray_r')

