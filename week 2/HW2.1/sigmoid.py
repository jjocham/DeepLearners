import numpy as np
from math import exp

class Sigmoid:  
    def __call__(self, inputs):
        return 1./(1.+ np.exp(-inputs))
    def backwards(self, preacts, activations, errors):
        dvector = activations*(1-activations)*errors
        jacobian = np.diag(dvector) # idk if that really works
        return dvector
    
# backward step confuses me it should be a diagonal matrix...
x = np.array([[-2,5,-3,3],[-4,2,-1,7],[-3,4,-3,2]])
errors = np.array([[-0.9, 0.2, 0.9, -0.5],[-2, -4, -3, 2],[1, 0.1, 1, 0.2]])
sig = Sigmoid()
y = sig(x)
print(y)
dx = sig.backwards(x,y,errors)
print(dx)
