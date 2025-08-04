import matplotlib.pyplot as plt
import numpy as np
import os
from plotting_widget import save_plot
import torch 
import torch.nn.functional as F
# for reproducibility
# torch.manual_seed(2025)
class RateRNN:
    def __init__(self, n_neurons, tau=10.0, dt=1.0,exc = 0.1, act=torch.tanh,lr = 1e-3,decay_r = 1e-4,threshold = 0,eta=1e-3,sigma=0.1):
        self.n_neurons = n_neurons
        self.tau = tau # rate constant 
        self.dt = dt # discretization step
        self.exc = torch.rand(n_neurons) # excitability 
        self.act = act # activation function
        self.lr = lr # learning rate for synaptic weights
        self.decay_r = decay_r #decay rate for synaptic weights
        self.threshold = threshold # activity threshold for plasticity 
        self.eta = eta
        self.sigma = sigma
        # Initialize random recurrent weights
        self.W = torch.randn(n_neurons, n_neurons) / n_neurons**0.5
        # Zero initial rate state
        self.rates = torch.zeros(n_neurons)

    def step(self,input_vector):
        """
        Perform one timestep of rate dynamics:
        input_vector: shape [n_neurons]
        """

        input_current = self.W @ self.rates + input_vector + self.exc
        dr_dt = (-self.rates + self.act(input_current)) / self.tau
        self.rates += (dr_dt*self.dt)

        post_mask = (self.rates > self.threshold).float()
        # weight increase due to habbian plasticity
        hebbian_dw = self.lr * torch.outer(self.rates*post_mask,self.rates) * self.dt

        # weight decay due to weight homeostasis
        decay = self.decay_r * self.W

        noise_update1 = self.generate_noise()
        noise_update2 = self.generate_noise()
        print(torch.mean(noise_update1/noise_update2))
        # print("mean signal to noise ratio = ",torch.mean(hebbian_dw/noise_update))
        self.W += hebbian_dw - decay + noise_update1

        # self.W = torch.clamp(self.W, -1.0, 1.0)

        return self.rates.detach().clone()
    
    def plot_weight_matrix(self,fname,title = "",xlab = "Pre Neurons",ylab = "Post Neurons",save_it = 1):
        plt.figure(figsize=(6,6))
        plt.imshow(self.W.cpu().detach().numpy(), cmap='bwr', interpolation='nearest')
        plt.colorbar(label='Weight value')
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        if save_it == 1:
            save_plot(fname)
        plt.show()
    def generate_noise(self):
        std_dev = (self.eta **0.5) * self.sigma
        noise_w = torch.rand(self.n_neurons,self.n_neurons) * std_dev
        return noise_w
n = 100
nn = RateRNN(n,tau=1000.0,dt = 1.0,lr = 0.1,threshold = 0.1)
timesteps = 200
input_arr = torch.abs(torch.randn(n))*0.1
input_arr[:1] += 5
nn.exc[:20] +- 2
print(input_arr)
op_dir = "./plots/torch_version"
FR_history = []
save_it = 0
nn.plot_weight_matrix(title = "Init. weights", fname = os.path.join(op_dir,"inital_weight"),save_it =save_it)
for t in range(timesteps):
    next_FR = nn.step(input_arr)
    FR_history.append(next_FR.numpy())

# Create input history as a matrix: shape (timesteps, n_neurons)
input_history = np.tile(input_arr.cpu().numpy(), (timesteps, 1))

# If you have varying input per timestep, replace input_history with that data

plt.figure(figsize=(10, 6))
plt.imshow(input_history.T, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Input value')
plt.ylabel('Neuron index')
plt.xlabel('Time step')
plt.title('Inputs to neurons over time')
plt.show()

timesteps_step2 = timesteps*10
input_arr[:1] -= 5
nn.plot_weight_matrix(title = "Final weights", fname = os.path.join(op_dir,"final_weight"),save_it =save_it )
for t in range(timesteps_step2):
    next_FR = nn.step(input_arr)
    FR_history.append(next_FR.numpy())

# Create input history as a matrix: shape (timesteps, n_neurons)
input_history2 = np.tile(input_arr.cpu().numpy(), (timesteps_step2, 1))

input_history = np.concat((input_history,input_history2))
# If you have varying input per timestep, replace input_history with that data

plt.figure(figsize=(10, 6))
plt.imshow(input_history.T, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Input value')
plt.ylabel('Neuron index')
plt.xlabel('Time step')
plt.title('Inputs to neurons over time')
plt.show()

FR_history = np.stack(FR_history)
plt.imshow(FR_history.T, aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('Time step')
plt.ylabel('Neuron')
plt.title('Population firing rates (heatmap)')
plt.colorbar(label='Firing rate')
plt.tight_layout()
plt.show()