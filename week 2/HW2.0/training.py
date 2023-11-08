from mlp import MLP
import numpy as np
import random as rand

# initiate target values: -----------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. Randomly generate 100 numbers between 0 and 1 and save them to an
# array ’x’. These are your input values.
x = [rand.random() for i in range(0,100)]
#print(len(x))

# 2. Create an array ’t’. For each entry x[i] in x, calculate x[i]**3-x[i]**2
# and save the results to t[i]. These are your targets.
t = np.zeros(100)

def f(x):
    return x**3-x**2

for index, entry in enumerate(x):
    t[index] = entry**3-entry**2

# create mlp: -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a MLP with 1 hidden layer consisting of 10 units which all
# receive a single input, and an output layer with just 1 unit.
def loss_func(y, t):
    return (1/2)*((y-t)**2)

average_loss = None

for epoch in range(1000):
    current_loss = []
    for i,d in enumerate(x):
        mlp = MLP(n_layers=2, input_vector=d, n_units=[10,1])
        loss = loss_func(mlp.forward_feeding(),t[i])
        mlp.backpropagation(loss=loss, learning_rate=0.04)
        current_loss.append(loss)
    average_loss.append(sum(current_loss)/ len(current_loss))

print(average_loss)




