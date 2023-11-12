import numpy as np
from math import exp

class Sigmoid:
    def __call__(self, inputs):
        """
            initialize an instance of the sigmoid function with
            input: of ndarray (preactivations) 
            returns: activations of current layer
        """
        activations = 1./(1.+ np.exp(-inputs))
        return activations
    
    def backwards(self, preacts, activations, errors):
        """
            performs the backwards-step, calculates the derivative of the sigmoid funciton with respect to preactivations 
            and applies it to the error signal
            inputs: ndarrays
            return: diagonal jacobian matrix of calculated sigmoid derivative 

        """
        jacobian = errors * activations * (1-activations)
        return jacobian
        

    