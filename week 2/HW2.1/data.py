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
def shuffle_generator(inputs, targets, minibatch_size):
    """
        arg: accepts specified minibatch-size, input and target values
        returns: shuffled input-target pairs in batches of size minibatch_sizex64 and minibatch_sizex10
    """
    if len(inputs)%minibatch_size != 0:
        raise ValueError("not a legal minibatchsize")
    # Generate a random permutation of indices
    indices = np.random.permutation(len(inputs))
    # suffle pairs with new minibatch indeces but keep pairs intact
    for start in range(0, len(inputs), minibatch_size):
        end = start+minibatch_size
        indices_of_batch = indices[start:end]
        yield inputs[indices_of_batch], targets[indices_of_batch]
