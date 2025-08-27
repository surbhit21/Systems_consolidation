import matplotlib.pyplot as plt
import numpy as np
import os
from plotting_widget import *
import torch 
import torch.nn.functional as F
from Utilities import average_firing_rates_with_active,ensamble_overlap
class TwoAreaModel:
    def __init__(self, n_inp, nMTL, nCTX,eMTL,eCTX,tau=10.0, dt=1.0, act=torch.relu, 
                 lr_MTL=0.45, decay_MTL=0.55, 
                 lr_CTX=0.06, decay_CTX=0,
                 threshold=0, I0=10, I1=0.8, I2=0.01):
        
        self.n_inp = n_inp  # number of input neurons
        self.MTL = nMTL
        self.CTX = nCTX
        self.tau = tau  # rate constant 
        self.dt = dt  # discretization step
        self.act = act  # activation function
        self.I0 = I0
        self.I1 = I1
        self.I2 = I2
        self.lr_MTL = lr_MTL  # learning rate
        self.decay_MTL = decay_MTL  # weight decay rate
        self.lr_CTX = lr_CTX  # learning rate
        self.decay_CTX = decay_CTX  # weight decay rate
        self.threshold = threshold
        self.ext_MTL = eMTL
        self.ext_CTX = eCTX
        self.g_ctx = 0.5
        # Zero initial rate state
        self.r_MTL = torch.zeros(nMTL)
        self.r_CTX = torch.zeros(nCTX)
        self.r_INP = torch.zeros(n_inp)

        # Random initial weights
        # self.W_inp_MTL = torch.abs(torch.normal(mean=0.0, std=1.0, size=(n_inp, nMTL))) * 0.1
        # self.W_inp_CTX = torch.abs(torch.normal(mean=0.0, std=1.0, size=(n_inp, nCTX))) * 0.1
        self.W_MTL_CTX = 0.5
        self.W_MTL_MTL = torch.zeros(size=(nMTL, nMTL))
        self.W_CTX_CTX = torch.zeros(size=(nCTX, nCTX))


    
    def step(self):
        """
        Perform one timestep of rate dynamics:
        input_vector: shape [n_inp]
        """
        # calculating the input to the CA1
        total_inp_MTL = self.r_INP + self.W_MTL_MTL @ self.r_MTL + self.ext_MTL

        # calculating the input to the CA3
        total_inp_CTX = self.r_INP  + self.W_CTX_CTX @ self.r_CTX  + self.W_MTL_CTX * self.r_MTL + self.ext_CTX
        
        # blanket inhibition to the RNN
        I_inhib_MTL = self.I0 + self.I1 * torch.sum(self.r_MTL) + self.I2 * torch.sum(self.r_MTL**2)

        # blanket inhibition to the RNN
        I_inhib_CTX = self.I0 + self.I1 * torch.sum(self.r_CTX) + self.I2 * torch.sum(self.r_CTX**2)


        # rate change as the nonlinear ODE
        dr_MTL_dt = (-self.r_MTL + self.act(total_inp_MTL - I_inhib_MTL)) / self.tau
        dr_CTX_dt = (-self.r_CTX + self.act(self.g_ctx*(total_inp_CTX - I_inhib_CTX))) / self.tau

        # updaiting the rates
        self.r_MTL += (dr_MTL_dt * self.dt)
        self.r_CTX += (dr_CTX_dt * self.dt)

        # Hebbian plasticity in weight changes
        post_mask_MTL = (self.r_MTL > self.threshold).float()
        post_mask_CTX = (self.r_CTX > self.threshold).float()

        hebbian_dW_MTL_MTL = self.lr_MTL * torch.outer(self.r_MTL*post_mask_MTL, self.r_MTL) * self.dt
        hebbian_dW_CTX_CTX = self.lr_CTX * torch.outer(self.r_CTX*post_mask_CTX, self.r_CTX) * self.dt
        # hebbian_dW_MTL_CTX = self.lr_CTX * torch.outer(self.r_CTX*post_mask_CTX, self.r_MTL) * self.dt

        # hebbian_dW_inp_MTL = self.lr_MTL * torch.outer(self.r_INP,self.r_MTL*post_mask_MTL) * self.dt
        # hebbian_dW_inp_CTX = self.lr_CTX * torch.outer(self.r_INP,self.r_CTX*post_mask_CTX) * self.dt

        # weight decay due to weight homeostasis
        decay_MTL_MTL = self.decay_MTL * self.W_MTL_MTL * self.dt
        decay_CTX_CTX = self.decay_CTX * self.W_CTX_CTX * self.dt
        decay_MTL_CTX = self.decay_CTX * self.W_MTL_CTX * self.dt
        # decay_inp_MTL = self.decay_MTL * self.W_inp_MTL * self.dt
        # decay_inp_CTX = self.decay_CTX * self.W_inp_CTX * self.dt
        # breakpoint()

        # applying the weight change
        self.W_MTL_MTL += (hebbian_dW_MTL_MTL - decay_MTL_MTL)
        self.W_CTX_CTX += (hebbian_dW_CTX_CTX - decay_CTX_CTX)
        # self.W_MTL_CTX += (hebbian_dW_MTL_CTX - decay_MTL_CTX)
        # self.W_inp_MTL += (hebbian_dW_inp_MTL - decay_inp_MTL)
        # self.W_inp_CTX += (hebbian_dW_inp_CTX - decay_inp_CTX)

        # Ensure weights are non-negative
        self.W_MTL_MTL = torch.clamp(self.W_MTL_MTL, 0, 1)
        self.W_CTX_CTX = torch.clamp(self.W_CTX_CTX, 0, 1)
        # self.W_MTL_CTX = torch.clamp(self.W_MTL_CTX, 0, 1)
        # self.W_inp_MTL = torch.clamp(self.W_inp_MTL, 0, 1)
        # self.W_inp_CTX = torch.clamp(self.W_inp_CTX, 0, 1)
       
        return self.r_MTL.detach().clone(),self.r_CTX.detach().clone()

torch.manual_seed(2025)
n_INP = 40
n_MTL = 40
n_CTX = 40
e_MTL = torch.zeros(size=(n_MTL,))#torch.abs(torch.normal(mean=0.0, std=1.0, size=(n_MTL,)))
e_MTL[:8] += 4
e_CTX = torch.zeros(size=(n_CTX,))#torch.abs(torch.normal(mean=0.0, std=1.0, size=(n_CTX,))) 
e_CTX[:8] += 3
act_threshold = 0.5
nn = TwoAreaModel(n_INP,n_MTL,n_CTX,e_MTL,e_CTX,tau=50.0,dt = 0.1,lr_MTL = 1/700,decay_MTL=1/1000,lr_CTX = 0,decay_CTX=1e-4,threshold=0)
t_FC = 2000
ID = 1000
input = 10*torch.ones(n_INP) 
zero_input = 0*input
# input[:] = 5
nn.r_INP = input
MTL_FR_history = []
CTX_FR_history = []
rec_weights_MTL = []
rec_weights_CTX = []
# ff_weights_MTL = []
# ff_weights_CTX = []
mtl_to_ctx_weights = []
ex_mtl_history = []
ex_ctx_history = []
for t in range(t_FC):
    next_MTL_FR,next_CTX_FR = nn.step()
    frs = (next_MTL_FR.numpy() > act_threshold)
    MTL_FR_history.append(next_MTL_FR.numpy() *frs)
    frs = (next_CTX_FR.numpy() > act_threshold)
    CTX_FR_history.append(next_CTX_FR.numpy() *frs)
    ex_mtl_history.append(nn.ext_MTL.detach().clone().numpy())
    ex_ctx_history.append(nn.ext_CTX.detach().clone().numpy())
input_history = np.tile(input, (t_FC, 1))

rec_weights_MTL.append(nn.W_MTL_MTL.detach().clone().numpy())
rec_weights_CTX.append(nn.W_CTX_CTX.detach().clone().numpy())
# ff_weights_MTL.append(nn.W_inp_MTL.detach().clone().numpy())
# ff_weights_CTX.append(nn.W_inp_CTX.detach().clone().numpy())
# mtl_to_ctx_weights.append(nn.W_MTL_CTX.detach().clone().numpy())
nn.r_INP = zero_input
for t in range(0,ID):
    next_MTL_FR,next_CTX_FR = nn.step()
    frs = (next_MTL_FR.numpy() > act_threshold)
    MTL_FR_history.append(next_MTL_FR.numpy() *frs)
    frs = (next_CTX_FR.numpy() > act_threshold)
    CTX_FR_history.append(next_CTX_FR.numpy() *frs)
    ex_mtl_history.append(nn.ext_MTL.detach().clone().numpy())
    ex_ctx_history.append(nn.ext_CTX.detach().clone().numpy())
input_at_t = np.tile(nn.r_INP, (ID, 1))
input_history = np.concatenate((input_history,input_at_t))
t_off = 200
IR_off = 100
# input = 20*torch.zeros(n_INP)
# nn.r_INP = input
num_days = 5
nn.W_MTL_CTX = 0
reps = 10
for day in range(num_days):
    nn.ext_MTL[day*8:(day+1)*8] -= 4
    nn.ext_MTL[(day+1)*8:(day+2)*8] += 4
    for i in range(reps):
        nn.r_INP = input
        for t in range(0,t_off):
            next_MTL_FR,next_CTX_FR = nn.step()
            frs = (next_MTL_FR.numpy() > act_threshold)
            MTL_FR_history.append(next_MTL_FR.numpy() *frs)
            frs = (next_CTX_FR.numpy() > act_threshold)
            CTX_FR_history.append(next_CTX_FR.numpy() *frs)
            ex_mtl_history.append(nn.ext_MTL.detach().clone().numpy())
            ex_ctx_history.append(nn.ext_CTX.detach().clone().numpy())
        input_at_t = np.tile(nn.r_INP, (t_off, 1))
        input_history = np.concatenate((input_history,input_at_t))

        nn.r_INP = zero_input
        for t in range(0,IR_off):
            next_MTL_FR,next_CTX_FR = nn.step()
            frs = (next_MTL_FR.numpy() > act_threshold)
            MTL_FR_history.append(next_MTL_FR.numpy() *frs)
            frs = (next_CTX_FR.numpy() > act_threshold)
            CTX_FR_history.append(next_CTX_FR.numpy() *frs)
            ex_mtl_history.append(nn.ext_MTL.detach().clone().numpy())
            ex_ctx_history.append(nn.ext_CTX.detach().clone().numpy())
        input_at_t = np.tile(nn.r_INP, (IR_off, 1))
        input_history = np.concatenate((input_history,input_at_t))

    rec_weights_MTL.append(nn.W_MTL_MTL.detach().clone().numpy())
    rec_weights_CTX.append(nn.W_CTX_CTX.detach().clone().numpy())

    nn.r_INP = zero_input
    for t in range(0,ID):
        next_MTL_FR,next_CTX_FR = nn.step()
        frs = (next_MTL_FR.numpy() > act_threshold)
        MTL_FR_history.append(next_MTL_FR.numpy() *frs)
        frs = (next_CTX_FR.numpy() > act_threshold)
        CTX_FR_history.append(next_CTX_FR.numpy() *frs)
        ex_mtl_history.append(nn.ext_MTL.detach().clone().numpy())
        ex_ctx_history.append(nn.ext_CTX.detach().clone().numpy())
    input_at_t = np.tile(nn.r_INP, (ID, 1))
    input_history = np.concatenate((input_history,input_at_t))
    
# ff_weights_MTL.append(nn.W_inp_MTL.detach().clone().numpy())
# ff_weights_CTX.append(nn.W_inp_CTX.detach().clone().numpy())
# mtl_to_ctx_weights.append(nn.W_MTL_CTX.detach().clone().numpy())


MTL_FR_history = np.array(MTL_FR_history)
CTX_FR_history = np.array(CTX_FR_history)
ex_mtl_history = np.array(ex_mtl_history)
ex_ctx_history = np.array(ex_ctx_history)
rec_weights_CTX = np.array(rec_weights_CTX)
rec_weights_MTL = np.array(rec_weights_MTL)
# ff_weights_CTX = np.array(ff_weights_CTX)
# ff_weights_MTL = np.array(ff_weights_MTL)
# mtl_to_ctx_weights = np.array(mtl_to_ctx_weights)
labs = ["After FC","of1","of2","of3","of4","of5"]
breakpoint()
plot_activity_n_excitability_time([MTL_FR_history.T,CTX_FR_history.T],
                       titles=['Neuronal Activity (MLT)',
                               'Neuronal Activity (CTX)'],
                       fname="./plots/twoRNN/Activity.png",
                       cmaps=['Blues','Blues'])
plot_activity_n_excitability_time([ex_mtl_history.T,ex_ctx_history.T,input_history.T],
                       titles=[
                                "Neuronal Excitability (MTL)",
                                "Neuronal Excitability (CTX)",
                                "Input FR"],
                       fname="./plots/twoRNN/excitability_n_input.png",
                       cmaps=[ 'Greens','Greens',"Reds"])

plot_weights_over_time(rec_weights_MTL,
                       titles= labs,
                       fname="./plots/twoRNN/Rec_w_MTL.png",
                       cmaps='gray_r')
plot_weights_over_time(rec_weights_CTX,
                       titles= labs,
                       fname="./plots/twoRNN/Rec_w_CTX.png",
                       cmaps='gray_r')
# plot_weights_over_time(ff_weights_MTL,
#                        titles= labs,
#                        fname="./plots/twoRNN/FF_w_MTL.png",
#                        cmaps='gray_r')
# plot_weights_over_time(ff_weights_CTX,
#                        titles= labs,
#                        fname="./plots/twoRNN/FF_w_CTX.png",
#                        cmaps='gray_r')
# plot_weights_over_time(mtl_to_ctx_weights,
#                        titles= labs,
#                        fname="./plots/twoRNN/MTL_to_CTX_w.png",
#                        cmaps='gray_r')  