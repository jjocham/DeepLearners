import numpy as np
from sigmoid import Sigmoid
from softmax import Softmax
from data import *

# Parameters
batch_size = 16
shuffled_data_generator = shuffle_generator(images_reshaped, onehot_targets, batch_size)
input_size = 64
output_size = 10


class MLP_layer():
    def __init__(self, num_inputs, num_units, activation_function, use_bias=True):
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.use_bias = use_bias
        self.weights = np.random.normal(loc=0.0, scale=0.2, size=(self.num_units, self.num_inputs))
        self.bias = np.zeros((self.num_units,))
        if activation_function == "sigmoid":
            self.act_function = Sigmoid()
        elif activation_function == "softmax":
            self.act_function = Softmax()

    # adjust weights
    def set_weights(self, weights, biases):
        self.weights = weights
        self.bias = biases

    def forward(self, x):
        if x.shape != (batch_size, self.num_inputs):
            raise AssertionError(f"The input should be of shape ({batch_size}, {self.num_inputs}) but you idiot gave me {x.shape}!?")
        pre_activations = self.weights @ x + np.transpose(self.bias)
        activations = self.act_function(pre_activations)
        return activations
    
    def weights_backward(self, d_preacts, preacts):
        d_weights = self.act_function.backwards(preacts)*np.sum(self.weights, axis=0)*d_preacts
        d_inputs = d_preacts
        return d_weights, d_inputs
    
    def backward():
        return

class MLP():
    def __init__(self, num_layers, num_units_for_layers, activation_functions):
        if num_layers != len(num_units_for_layers)-1:
            raise AssertionError(f"You have to specify as many num_units as you got layers!!")
        if set(activation_functions) != {"sigmoid", "softmax"} and len(activation_functions) == num_layers:
            raise AssertionError(f"You have to specify as many activation_functions as you got layers and only use 'softmax' or 'sigmoid'")
        self.layer_list = [MLP_layer(num_units_for_layers[l], num_units_for_layers[l+1], activation_functions[l]) for l in range(num_layers)]

# play around with different layers and functions
my_cute_mlp = MLP(2, [input_size,16,output_size], ["sigmoid", "softmax"])

# Training epochs
for layer in my_cute_mlp:
    layer.forward()
