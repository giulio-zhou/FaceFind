import numpy as np

# Dimensions of the training set images in pixels
ts_width = 92
ts_height = 92

# Length of the feature vector of an image
fv_length = ts_height * ts_width

# Number of hidden nodes
nn_hidden = fv_length//4

# Number of output nodes
nn_output = 2

# The positive and negative vectors, respectively.
pos_vec = np.array([1, 0]).T
neg_vec = np.array([0, 1]).T

# Sliding window step sizes
xstep = 5
ystep = 5
