# Create a MLP class which combines instances of your Layer class into a
# MLP. Implement two methods:
# 1. A forward_step method which passes an input through the entire network
# 2. A backpropagation method which updates all the weights and biases in
# the network given a loss value.

from layer import Layer
import numpy as np

class MLP:
    """
    combines instances of "Layer" class into a MLP

    """
    def __init__(self, n_layers, input_vector, n_units):

        # check for necessery requirements in specified arguments
        if len(n_units) != n_layers:
                 raise ValueError("you have to specify the number of units for the exact number of layers you want to initiate")
        
        self.n_layers = n_layers
        self.input_vector = np.array([input_vector])

        layers = []
        # instantiate a list of empty layers with length of n_layers
        for i in range(self.n_layers):
            if i == 0:
                layers.append(Layer(n_units=n_units[i], input_units=len(self.input_vector)))
            else: 
                layers.append(Layer(n_units[i], n_units[i-1]))

        self.layers = layers


    def forward_feeding(self):
        # instantiate output of input layer
        current_output = self.input_vector

        for layer in self.layers:
                # instantiates input of current layer (first time with input vector, then with output of prev layer)
                setattr(layer, "layer_input", current_output)
                # updates input for next layer and instantiates the layer_preactivation and layer_activation attributes of current layer
                current_output = layer.forward_step()
        
        # at the end "current input" should have the value of the output of last layer
        return current_output
    
    def backpropagation(self, loss, learning_rate):
        # instantiate gradient
        current_gradient = loss

        for layer in self.layers:
            # weights are updated when backward_step is being called
            # new gradient is output gradient of proceeding layer
            print("current gradient", current_gradient)
            current_gradient = layer.backward_step(current_gradient, learning_rate)
            
              
         
         
         







        




