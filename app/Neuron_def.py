import ANNarchy as ann
import numpy as np

class RatemodelNeuron():
    def __init__(self, para_dict, eqns):
        self.neuron = ann.Neuron(
        parameters = para_dict,
        equations = eqns
    ) 

class SynapseForRatemodel():
    def __init__(self, para_dict, eqns):
        self.synapse = ann.Synapse(
        parameters = para_dict,
        equations = eqns
    )