# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TASK 2.2
# implement a simple layer using perceptrons - layer should be fully connected - as you need several layers for the MLP, write a Layer class 
# Cloass should have:
# 1. A constructor 
# 2. A method called forward_step
# 3. A method called backward_step
import numpy as np 

class Layer: 

    def __init__(self, n_units, input_units):
        self.n_units = n_units
        self.input_units = input_units

        # Initialize weights with random values and biases with zeros
        self.weights = np.random.rand(input_units, n_units)
        self.biases = np.zeros(n_units)

        # Instantiate empty attributes for layer input, preactivation, and activation
        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None

    # A method called ’forward_step’, which returns each unit’s activation
    #(i.e. output) using ReLu as the activation function.
    def forward_step(self): # do I need n_units and input_units?:
            
            self.layer_preactivation = np.dot(self.layer_input,self.weights)+self.biases
            print("preactivation: ",self.layer_preactivation)
            self.layer_activation = [max(0, i) for i in self.layer_preactivation]
            return np.asarray(self.layer_activation)

    def backward_step(self, gradients, learning_rate):


           if self.layer_preactivation is None:
                 raise ValueError("Caution! preactivation values are empty")
           
           if self.layer_activation is None:
                 raise ValueError("Caution! activation values are empty")
           
           if self.layer_input is None:
                 raise ValueError("Caution! input values are empty")
           
           # First we need to compute the derivative of ReLU activation function
           relu_grad = (self.layer_preactivation > 0).astype(float) #returns zero for values <= 0 and else 1
           print("relu_grad",relu_grad) 
           print("self.layer_input.T", self.layer_input.T)
         
           # then we need partial derivatives with respect to the layer's weights
           deriv_w = np.dot(self.layer_input.T, (relu_grad*gradients))
           # now update the weights
           self.weights = self.weights - learning_rate*deriv_w
           # derivative with respect to bias
           deriv_b = relu_grad*gradients
           # now update bias
           self.biases = self.biases - learning_rate*np.sum(deriv_b, axis=0)

           # Lastly we need the gradient with respect to the layer's input, i.e. the error signal of the current layer
           # this is needed for the gradient computations in the preceeding layers
           deriv_input = np.dot(relu_grad*gradients, self.weights.T)
           # return gradient of input for preceeding layer
           return deriv_input

           



    
