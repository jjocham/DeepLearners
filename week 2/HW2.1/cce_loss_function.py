#cce_loss_function.py

import numpy as np
from softmax import Softmax

class Categorical_Cross_Entropy_Loss:
    def __call__(self, inputs, targets, minibatch_size, num_categories):
        if inputs.shape != (minibatch_size, num_categories):
            raise AssertionError(f"The inputs should be of shape ({minibatch_size}, {num_categories}) but you idiot gave me {inputs.shape}!?")
        if targets.shape != (minibatch_size, num_categories):
            raise AssertionError(f"The targets should be of shape ({minibatch_size}, {num_categories}) but you idiot gave me {targets.shape}!?")
        soft_function = Softmax()
        soften = soft_function(inputs)
        cce_loss = -sum(targets*np.log(soften))
        return cce_loss
    
    # in HA steht prediction and loss aber laut link ist es target
    def backwards(self, predictions, targets):
        jacobian = predictions - targets
        return jacobian
    
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
