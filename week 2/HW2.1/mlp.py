from math import exp
import numpy as np
from sigmoid import Sigmoid
from softmax import Softmax
from data import shuffle_generator
from sklearn.datasets import load_digits
from cce_loss_function import Categorical_Cross_Entropy_Loss

#data
digits = load_digits()
x = digits.data
y = digits.target

#inputs
x = x/np.max(x)
x = np.float32(x)

#target
y = np.eye(10, dtype=int)[y]

#delete last 3 rows
x = x[:-3]
y = y[:-3]

# Parameters
batch_size = 13
shuffled_data_generator = shuffle_generator(x, y, batch_size)
input_size = 64
output_size = 10



class MLP_layer():
    def __init__(self, num_inputs, num_units, activation_function, use_bias=True):
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.use_bias = use_bias
        self.weights = np.random.normal(loc=0.0, scale=0.2, size=(self.num_inputs, self.num_units ))
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
        #pre_activations = self.weights @ x + np.transpose(self.bias)
        pre_activations = np.dot(x, self.weights) + np.transpose(self.bias)
        activations = self.act_function(pre_activations)
        
        return activations
    
    def weights_backward(self, d_preacts, preacts):
        d_weights = self.act_function.backwards(preacts)*np.sum(self.weights, axis=0)*d_preacts
        d_inputs = d_preacts
        return d_weights, d_inputs
    
    def backward(self, error_signal, inputs):
        dL_dW = np.dot(inputs.T, error_signal)
        dL_dinput = np.dot(error_signal, self.weights.T)
        
        return dL_dW, dL_dinput

class MLP():
    def __init__(self, num_layers, num_units_for_layers, activation_functions):
        if num_layers != len(num_units_for_layers)-1:
            raise AssertionError(f"You have to specify as many num_units as you got layers!!")
        if set(activation_functions) != {"sigmoid", 'softmax'} and len(activation_functions) == num_layers:
            raise AssertionError(f"You have to specify as many activation_functions as you got layers and only use 'softmax' or 'sigmoid'")
        self.layer_list = [MLP_layer(num_units_for_layers[l], num_units_for_layers[l+1], activation_functions[l]) for l in range(num_layers)]
        
    def forward(self, x, target):
        for i in range(len(self.layer_list)):
            x = self.layer_list[i].forward(x)
            
        func = Categorical_Cross_Entropy_Loss()
        loss = func(x,target)
        
        print(loss)
        return loss
    
    def get_data_forward(self, data_generator):
        
        batches = []
        
        while True:
            try:
                batches.append(next(shuffled_data_generator))
                
            except StopIteration:
                break
                
        for i in range(len(batches)):
            self.forward(batches[i][0], batches[i][1])
            
    def backward(self, x, y):
        
        layer_info = [{} for _ in range(len(self.layer_list))]
    
    
#create MLP
my_cute_mlp = MLP(2, [input_size,16,output_size], ["sigmoid", "softmax"])

#use MLP forward
my_cute_mlp.get_data_forward(shuffled_data_generator)


