import numpy as np
from math import exp

class Sigmoid: 
        
    def call(self, inputs):
      return 1./(1.+ np.exp(-inputs))
    
x = np.array([[4,5,3,3],[4,2,1,7],[8,5,6,2]])
sig = Sigmoid()
print(sig.call(x))