
import numpy as np
# Using a single line of code, get a list of the squares
# of each number between 0 and 100! Then do it again, but only include those
# squares which are even numbers.


list = [pow(x,2) for x in list(range(0,100)) if x%2 == 0]

print(list)