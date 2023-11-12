import numpy as np

class Softmax:
    def __call__(self, inputs):
        """
            initialize an instance of the softmax funtion 
            input: of ndarray
            returns: a probability distribution with respect to the activations of the last layer
        """
        # calculation of softmax
        inputs_exp = np.exp(inputs)
        sum_exp = np.sum(inputs_exp, axis=1, keepdims=True)
        return inputs_exp/sum_exp


