import numpy as np

class Softmax:
    def __call__(self, inputs):
        inputs_exp = np.exp(inputs)
        sum_exp = np.sum(inputs_exp, axis=1, keepdims=True)
        return inputs_exp/sum_exp

inputs = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

output = Softmax()
dist = output(inputs)
# print("nerv nicht")
# print(dist)
# print(np.sum(dist, axis=1))
