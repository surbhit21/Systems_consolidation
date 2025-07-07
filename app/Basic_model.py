import numpy as np
import matplotlib.pyplot as plt

class RateNetwork():
    """
    A class to represent a rate network model.
    """
    def __init__(self, num_neurons, init_weight):
        """
        Initialize the rate network with a specified number of neurons and inputs.
        
        Parameters:
        num_neurons (int): Number of neurons in the network.
        num_inputs (int): Number of input features.
        """
        self.num_neurons = num_neurons
        self.weights = init_weight  # Random weights initialization
       
    def act_function(self,x):
        """
        tanh activation function
        """
        return np.tanh(x)

    def grad(self, x,W,I_ext):
        """
        Forward pass through the network.
        
        Parameters:
        I_ext (np.ndarray): External input to the network.
        
        Returns:
        np.ndarray: Output of the network after applying activation function.
        """
        # Compute the weighted sum of inputs
        z = self.act_function(W @ x) + I_ext

        
        # Apply activation function and calculating gradient
        dxdt = (- x + z)/10
        
        return dxdt
    
    def update_weights(self, W,x,learning_rate, decay_rate):
        firing_rates = self.act_function(x)
        # firing_rates[np.abs(firing_rates) < 0.4] = 0

        # dWdt = (learning_rate * (1-W) * np.outer(firing_rates, firing_rates)  - decay_rate * W)
        dWdt = 0
        return dWdt

    def run_sim(self, x_init,I_ext, learning_rate=0.01, decay_rate=0.001, num_steps=1000,dt=0.1):
        """
        Run the simulation of the rate network.
        
        Parameters:
        I_ext (np.ndarray): External input to the network.
        learning_rate (float): Learning rate for weight updates.
        decay_rate (float): Decay rate for weight updates.
        num_steps (int): Number of simulation steps.
        
        Returns:
        np.ndarray: Outputs of the network at each step.
        """
        outputs = []
        x = np.copy(x_init)
        for step in range(1,num_steps):
            x += dt*self.grad(x,self.weights,I_ext[step])
            outputs.append(x)
            dWdt = self.update_weights(self.weights, x, learning_rate, decay_rate)
            # print(dWdt)
            self.weights += (dWdt*dt)
        
        return np.array(outputs)

num_neurons = 100
time_steps = 10000
dt = 0.1
init_weight = 1.5* np.random.rand(num_neurons, num_neurons) / np.sqrt(num_neurons)  # Initialize weights with small random values
I_ext = np.zeros((time_steps, num_neurons))
I_ext[:, 0] = 10  # External input to the first neuron
x = np.random.rand(num_neurons)  # Initial state of the neurons
RN = RateNetwork(num_neurons=num_neurons, init_weight=init_weight)
outputs = RN.run_sim(x_init=x,I_ext=I_ext, learning_rate=0.00001, decay_rate=0, num_steps=time_steps,dt=dt)
# breakpoint()
# Plotting the outputs
plt.figure(figsize=(10, 6))
plt.plot(outputs[:,0:10])
plt.title('Outputs of the Rate Network over Time')
plt.xlabel('Time Steps')
plt.ylabel('Neuron Outputs')
plt.legend([f'Neuron {i}' for i in range(num_neurons)])
# plt.grid()
plt.show()
# The code defines a rate network model, simulates its behavior over time, and visualizes the outputs of the neurons.

