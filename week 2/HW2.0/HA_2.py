# 2.01 - Building your Dataset



# Optional: Plot your data points along with the underlying function which
# generated them.

import numpy as np
import random as rand
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TASK 2.1

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

# Optional: Plot your data points along with the underlying function which
# generated them.

import matplotlib.pyplot as plt

xpoints = np.arange(1,101)

#plt.plot(xpoints, t)
#plt.plot(xpoints, f(xpoints), color="red")
#plt.show()
#print(np.shape(t))
#print(xpoints)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TASK 2.2
# implement a simple layer using perceptrons - layer should be fully connected - as you need several layers for the MLP, write a Layer class 
# Cloass should have:
# 1. A constructor 
# 2. A method called forward_step
# 3. A method called backward_step

x = np.array([1])
print(len(x))
