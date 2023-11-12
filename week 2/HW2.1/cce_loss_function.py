import numpy as np
from softmax import Softmax

class Categorical_Cross_Entropy_Loss:
    def __call__(self, inputs, targets):
        """
            calculates the difference between target and MLP predictions 
            input: predictions (ndarray), given targets (ndarray)
            returns: the cce loss difference
        """
        # uses the softmax activation function
        soft_function = Softmax()
        soften = soft_function(inputs)
        # calculates cce loss
        cce_loss = -np.sum(targets*np.log(soften), axis = 1)
        return cce_loss
    
    def backwards(self, predictions, targets):
        """
            is the backwards function of the softmax and the cce loss funciton at once
            returns: derivative with respect to softmax and cce
        """
        derivative = predictions - targets
        return derivative
    
# x = np.array([[5,5,-3,3,2],[-2,3,8,2,3],[-3,-4,3,-2,4]])
# targ = np.array([[1, 0, 0, 0,0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0]])
# cat = Categorical_Cross_Entropy_Loss()
# y = cat(x,targ,3,5)
# print("x.shape" + str(x.shape))
# print("y:")
# print(y)
# dx = cat.backwards(y, targ)
# print("dx:")
# print(dx)
