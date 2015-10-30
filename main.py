import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import skimage
import skimage.io as skio
import sys

from matplotlib.patches import Rectangle
from neural_network.neural_net import Neural_net
from neural_network.constants  import ts_width, ts_height, fv_length, nn_hidden
from neural_network.constants  import nn_output, pos_vec, neg_vec
from skin_detection.detect_skin import skinDetection
from skimage.transform import resize
from skimage.color import rgb2gray

parser = argparse.ArgumentParser(
    description="Train and classify with a fully-functional Neural Network")
parser.add_argument('--classify', action='store_true', default=False,
                    help="Include flag to use for classification"
                         "(default: train the neural network)")
parser.add_argument('--num_iter', default=100000, type=int, dest='num_iter',
                    help="The number of iterations to train (default: 100000)")
parser.add_argument('--reuse_weights', action='store_true', default=False,
                    help="Include flag to reuse old weights for training")
parser.add_argument('-i', '--img_path', help="Path to image/directory of images"
                                             "to classify")
parser.add_argument('-n', '--negative_ex', default='data/negative_examples',
                    help="Path to directory of negative training examples")
parser.add_argument('-p', '--positive_ex', default='data/positive_examples',
                    help="Path to directory of positive  training examples")
parser.add_argument('-v', '--visualize', action='store_true', default=False,
                    help="Include flag to visualize the preprocessing results")
args = parser.parse_args()


# Load old weights if specified and reinitialize if necessary
if args.reuse_weights:
    print("Reusing pre-trained weights!")
    if not (os.path.isfile('weights/theta1.npy') and
            os.path.isfile('weights/theta2.npy')):
        print("Cached weights not found. Exiting program...")
        sys.exit(1)
    theta1 = np.load(open('weights/theta1.npy', 'r'))
    theta2 = np.load(open('weights/theta2.npy', 'r'))
else:
    # Generate all weights randomly between -0.5 and 0.5
    theta1 = np.random.rand(fv_length + 1, nn_hidden) - 0.5
    theta2 = np.random.rand(nn_hidden + 1, nn_output) - 0.5

if args.classify:
    # Read in image or images, run preprocessing and perform detection
    if not args.img_path: 
        print("To perform classification, please provide a file/directory path")
        sys.exit(1)
    imgs_to_classify = []
    if os.path.isfile(args.img_path):
        # Perform classification on one input file
        input_img = skio.imread(args.img_path)
        imgs_to_classify.append(input_img)
    else:
        # Perform classification on all files in input directory
        filenames = os.listdir(args.img_path)
        for filename in filenames:
            img = skio.imread(args.img_path)
            imgs_to_classify.append(img)

    # Initialize the neural network with given parameters
    nn = Neural_net(fv_length, nn_hidden, nn_output, theta1, theta2)

    for img in imgs_to_classify:
        # Perform skin detection to obtain a set of bounding boxes
        bounding_box_imgs = skinDetection(img, show_img=args.visualize)

        # Draw the image and the resulting bounding boxes
        plt.imshow(img)
        for candidate_face, coordinates in bounding_box_imgs: 
            # Convert to grayscale and reshape for neural network use
            candidate_face = rgb2gray(candidate_face)
            candidate_face = resize(candidate_face, (ts_height, ts_width))
            candidate_face = candidate_face.reshape(ts_height*ts_width)
            result = nn.predict_one(candidate_face)
            top_left, bottom_right = coordinates

            # Draw rectangle if the object found is considered a face
            if np.argmax(result) == 0:
                current_axis = plt.gca()
                current_axis.add_patch(
                    Rectangle((top_left[0], bottom_right[1]),
                              bottom_right[0] - top_left[0],
                              top_left[1] - bottom_right[1],
                              fill=False, color='b'))

        # Show the result
        plt.show()
else:
    # Load in positive and negative training examples and train the neural net
    positive_examples = []
    negative_examples = []
    labels = []
    for filename in os.listdir(args.positive_ex):
        img = skio.imread('data/positive_examples/' + filename)
        img_vector = img.reshape(fv_length)
        positive_examples.append(img_vector)
        labels.append(pos_vec)

    for filename in os.listdir(args.negative_ex):
        img = skio.imread('data/negative_examples/' + filename)
        img_vector = img.reshape(fv_length)
        negative_examples.append(img_vector)
        labels.append(neg_vec)

    input_data = np.array(positive_examples + negative_examples)
    input_data_labels = np.array(labels)
    # Pass the data and label array to the neural network and train
    nn = Neural_net(fv_length, nn_hidden, nn_output, theta1, theta2)
    nn.train(input_data, input_data_labels, args.num_iter)

