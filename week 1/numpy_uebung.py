#Create a 5x5 NumPy array filled with normally distributed (i.e. µ = 0,
#σ = 1)
#2. If the value of an entry is greater than 0.09, replace it with its square.
#Else, replace it with 42.
#3. Use slicing to print just the fourth column of your array.

import numpy as np

#1.
arr = np.random.normal(loc=0.0, scale=1.0, size=(5,5))
print(arr)

#list = [pow(x,2) if x > 0.09 else 42 for x in arr]

for row in range(arr.shape[0]):
    for el in range(arr.shape[1]):
        if arr[row,el] > 0.09:
            arr[row, el] = round(pow(arr[row, el],2),5)
        else:
            arr[row, el] = 42.0

print(arr)

print(arr[:,3])