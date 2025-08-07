import matplotlib.pyplot as plt
import numpy as np
from plotting_widget import save_plot
import torch

class RateRNN_EI:
    def __init__(self, n_exc, n_inh, tau=10.0, dt=0.1, lr = 1e-2, phi=torch.relu,ex0 = 0.0,tau_e = 100, device='cpu'):
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        self.tau = tau
        self.dt = dt
        self.phi = phi
        self.device = device
        self.lr = lr
        self.excitability = torch.abs(torch.randn(n_exc + n_inh, device=device))  # Excitability for all neurons
        self.ex0 = ex0
        self.tau_e = tau_e
        # Weight matrix: initialize block structure
        # EE, EI: positive; IE, II: negative
        W = torch.zeros(self.n_total, self.n_total, device=device) #/ self.n_total**0.5
        W[n_exc:, n_exc:] = -torch.abs(torch.randn(n_inh, n_inh, device=device)) / n_inh**0.5  # II
        # Set blocks
        # Excitatory outgoing
        W[:n_exc, :] = torch.abs(W[:n_exc, :])       # E⭢ all, positive
        # Inhibitory outgoing
        W[n_exc:, :] = -torch.abs(W[n_exc:, :])      # I⭢ all, negative

        self.W = W
        self.rates = torch.zeros(self.n_total, device=device)

    def step(self, input_vector,target,plast_threshold=0.0):
    #    # rate dynamics
        # get total input to neurons
        input_current = torch.matmul(self.W, self.rates) + input_vector + self.excitability
        # change in rates
        dr_dt = (-self.rates + self.phi(input_current)) / self.tau
        self.rates = self.rates + self.dt * dr_dt
    
       
        # Excitatory plasticity
        post_mask = (self.rates > plast_threshold).float()  # Activity threshold
    #    plastiicty
    #    plasticity of E => All connections
        
        dw = torch.zeros_like(self.W, device=self.device)
        dw[:self.n_exc,:] = self.lr* self.dt * torch.outer(self.rates[:self.n_exc] * post_mask[:self.n_exc], self.rates) #hebbian plasticity in excitatory connections
        # breakpocint()
        # self.W += dw_e
    #    plasticity of I => E conncetions
        inh_pre = self.rates[self.n_exc:]
        exc_post = self.rates[:self.n_exc]
        hebb_i = torch.outer(inh_pre, exc_post)
        depp_i = target*inh_pre[:,None]
        
        # Get current weights I→E
        # w_inh_to_exc = self.W[self.n_exc:, :self.n_exc]

        # Update only I→E weights
        dw[self.n_exc:, :self.n_exc] =  self.lr*(hebb_i - depp_i)
        # w_inh_to_exc += dw
        self.W += dw
        self.W[:self.n_exc,:] = torch.clamp( self.W[:self.n_exc,:], min=0.0,max=1.0)
        # Clamp for biological plausibility (keep inhibitory synapses ≤ 0)
        self.W[self.n_exc:, :self.n_exc] = torch.clamp(self.W[self.n_exc:, :self.n_exc], max=0.0,min = -1.0)
        # breakpoint()
    #    excitability dynamics
        print(torch.max(torch.sum(dw,axis=0)))
        de_dt = (-self.excitability + torch.sum(dw,axis=1)) / self.tau_e
        self.excitability += (de_dt*self.dt)

        return self.rates.detach().clone()

    def excitability_increase(self,index_list,target):
        for x in index_list:
            self.excitability[x] += target

# Suppose W is your full weight matrix of shape (n_total, n_total)
# with n_exc excitatory neurons and n_inh inhibitory neurons
# Partition W into four blocks:
# E→E = W[:n_exc, :n_exc]
# E→I = W[:n_exc, n_exc:]
# I→E = W[n_exc:, :n_exc]
# I→I = W[n_exc:, n_exc:]

def plot_W_blocks(W, ne, ni):
    EE = W[:ne, :ne]
    EI = W[:ne, ne:]
    IE = W[ne:, :ne]
    II = W[ne:, ne:]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    im0 = axs[0, 0].imshow(EE, aspect='auto', cmap='Grays')
    axs[0, 0].set_title('E→E')
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im1 = axs[0, 1].imshow(EI, aspect='auto', cmap='Grays')
    axs[0, 1].set_title('E→I')
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im2 = axs[1, 0].imshow(IE, aspect='auto', cmap='Grays')
    axs[1, 0].set_title('I→E')
    fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im3 = axs[1, 1].imshow(II, aspect='auto', cmap='Grays')
    axs[1, 1].set_title('I→I')
    fig.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04)

    for ax in axs.flat:
        ax.set_xlabel('Post-synaptic neuron')
        ax.set_ylabel('Pre-synaptic neuron')

    plt.tight_layout()
    plt.show()



def plot_W_with_blocks(W, n_exc, n_inh):
    W_np = W  # convert to numpy for plotting
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(W_np, cmap='Grays', aspect='auto')

    # Draw lines to mark the E/I boundary
    ax.axhline(n_exc - 0.5, color='red', linewidth=2)
    ax.axvline(n_exc - 0.5, color='red', linewidth=2)
    
    # Label the four blocks
    ax.text(n_exc/2, n_exc/2, 'E→E', color='red', fontsize=14, ha='center', va='center', weight='bold')
    ax.text(n_exc + n_inh/2, n_exc/2, 'E→I', color='red', fontsize=14, ha='center', va='center', weight='bold')
    ax.text(n_exc/2, n_exc + n_inh/2, 'I→E', color='red', fontsize=14, ha='center', va='center', weight='bold')
    ax.text(n_exc + n_inh/2, n_exc + n_inh/2, 'I→I', color='red', fontsize=14, ha='center', va='center', weight='bold')
    
    ax.set_xlabel('Post-synaptic neuron')
    ax.set_ylabel('Pre-synaptic neuron')
    ax.set_title('Weight matrix W with E/I blocks')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Weight value')
    plt.show()

# Example usage:
# Assuming W, n_exc, n_inh are defined
# plot_W_with_blocks(W, n_exc, n_inh)

# Example usage:
# Assuming you have W, n_exc, n_inh defined as in your model
# plot_W_blocks(W, n_exc, n_inh)

n_exc = 400  # e.g., 80% excitatory
n_inh = 100 # e.g., 20% inhibitory
rnn_ei = RateRNN_EI(n_exc, n_inh, device='cpu',tau=100)
input_signal = torch.zeros(n_exc + n_inh)
input_signal[:50] = 10  # Excitatory input to all excitatory neurons
input_signal[n_exc:] = 10
FR_history = []
excitability_history = []
timesteps = 200
target_FR = 0.1
weight_matrix = []
# rnn_ei.step(input_signal)
weight_matrix.append(rnn_ei.W.detach().clone().cpu().numpy())

for t in range(timesteps):
    # input_vector = torch.randn(n)  # Random input for demonstration
    next_FR = rnn_ei.step(input_signal,target_FR)
    FR_history.append(next_FR.numpy())
    excitability_history.append(rnn_ei.excitability.detach().clone().cpu().numpy())
# Example: Suppose activity_matrix is your (n_neurons, n_timesteps) array
# For demonstration, here is a toy example:
# activity_matrix = np.random.rand(n_neurons, n_timesteps)



weight_matrix.append(rnn_ei.W.detach().clone().cpu().numpy())
breakpoint()
FR_history = np.stack(FR_history)
plt.figure(figsize=(10, 6))
plt.imshow(FR_history.T, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Activity (firing rate)')
plt.xlabel('Time step')
plt.ylabel('Neuron index')
plt.title('Neuron activity matrix over time')
plt.show()

excitability_history = np.stack(excitability_history)
plt.figure(figsize=(10, 6))
plt.imshow(excitability_history.T, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Neuronal Excitability')
plt.xlabel('Time step')
plt.ylabel('Neuron index')
plt.title('Neuron excitability matrix over time')
plt.show()

plot_W_blocks(weight_matrix[1], n_exc, n_inh)
plot_W_with_blocks(weight_matrix[1], n_exc, n_inh)
# plt.figure(figsize=(10, 6))
# plt.imshow(weight_matrix[1], aspect='auto', origin='lower', cmap='Grays')
# plt.colorbar(label='Weight value')
# plt.xlabel('Time step')
# plt.ylabel('Neuron index')
# plt.title('Neuron activity matrix over time')
# plt.show()