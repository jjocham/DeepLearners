import numpy as np
#--------------------------------------DATA PREPROCESSING------------------------------------
# import data-set of handwritten digtis
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

# 1.  Get the data (images) and target values
images = digits.images
targets = digits.target

# Create a subplot to display multiple images 
fig, axes = plt.subplots(2,5, figsize=(10,4))
plt.suptitle("Example Digits from load_digits")

# 2. Make sure your input images seem correct by plotting them as images respectively!
for i, ax in enumerate(axes.ravel()):
    ax.imshow(images[i], cmap=plt.cm.gray)
    ax.set_title(f"Digit {targets[i]}")
    ax.axis("off")

plt.show()

# 3. Reshape the images into (64) vectors
images_reshaped = images.reshape(len(images), 64)

# 4.  Make sure the images are represented as float32 values within either the
# [0 to 1] or [-1 to 1] range, if necessary rescale them respectively
# Documentation says that pixel values in original data set lay between 0-16, 
# so we have to divide by 16 to normalize
images_reshaped = np.float32(images_reshaped)/16

# 5. One-hot encode the target digits 
onehot_targets = np.zeros((len(targets),10))
for i,e in enumerate(targets):
    onehot_targets[i][e]=1

# 6. Write a generator function, which shuffles the (input, target) pairs
def shuffle_generator(inputs, targets):

    # Generate a random permutation of indices
    indices = np.random.permutation(len(inputs))
    # suffle pairs with new index but keep pairs intact
    for index in indices:
        yield inputs[index], targets[index]

# 6. Write a generator function, which shuffles the (input, target) pairs
def shuffle_generator2():
    
    # Generate a random permutation of indices
    indices = np.random.permutation(len(inputs))
    # suffle pairs with new index but keep pairs intact
    for index in indices:
        yield inputs[index], targets[index]

# Create a generator using the shuffle_data_generator function
# Adjust your generator function to create minibatches: Combine minibatchsize many inputs into a ndarray of shape minibatch size, 64, and targets
# into a ndarray of shape minibatch size, 10 respectively. Make sure you
# can adjust your minibatchsize as an argument to this generator, and also
# that respective (input-target) pairs match with respect to their index in
# the minibatch
shuffled_data_generator = shuffle_generator(images_reshaped, onehot_targets)

