import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy import ndimage as nd
from scipy import misc
from scipy import ndimage
from scipy.ndimage import morphology
from skimage import data
from skimage.color import rgb2gray
from skimage.filter import gabor_kernel, canny, roberts
from skimage.measure import label
from skimage.morphology import remove_small_objects
from PIL import Image, ImageDraw

def skinDetection(img, show_img=False):
    def visualize(img, show_img):
        if show_img:
            plt.imshow(img, cmap=plt.cm.gray)
            plt.colorbar()
            plt.show()

    gray_img = rgb2gray(img)
    filtered = yCbCr_to_bin(img, 161.9964, -11.1051, 22.9265, 25.9997, \
                                 4.3568, 3.9479, 2)
    filtered = morphology.binary_fill_holes(filtered)
    visualize(filtered, show_img)

    # Remove areas smaller than threshold
    # filtered = morphology.binary_opening(filtered, structure=np.ones( (15, 15) ))
    filtered = remove_small_objects(filtered, 170)
    visualize(filtered, show_img)

    # First erosion
    filtered = morphology.binary_erosion(filtered, np.ones( (6, 6) ))
    visualize(filtered, show_img)

    visualize(gray_img, show_img)

    # Run canny edge detection on gray image
    # edge_img = canny(gray_img)
    edge_img = roberts(gray_img)
    edge_idx = np.where(edge_img > 0.1)
    edge_img = np.zeros(edge_img.shape, dtype=bool)
    edge_img[edge_idx] = True
    visualize(edge_img, show_img)

    # Combine edge detection result and filtered image to create composite
    edge_img = np.invert(edge_img)
    composite = np.logical_and(edge_img, filtered)
    visualize(composite, show_img)

    composite = morphology.binary_erosion(composite)
    visualize(composite, show_img)

    # Fill small holes
    composite = morphology.binary_fill_holes(composite)
    visualize(composite, show_img)

    # composite = morphology.binary_opening(composite, structure=np.ones( (3, 3) ))
    # composite = remove_area(composite, 170)
    composite = remove_small_objects(composite, 170)
    visualize(composite, show_img)

    # Label all connected components
    segments, num_segments = label(composite, background=0, return_num=True)
    
    # Generate square windows to encapsulate regions
    img_windows = []
    for i in range(num_segments): 
        segment_idx = np.where(segments == i)
        box_info = gen_box(segment_idx)
        img_windows.append(box_info)
    
    corners = []
    for box in img_windows:
        (x, y), width = box
        corner = [(x - width//2, y - width//2), (x + width//2, y + width//2)]
        corners.append(corner)

    # Extract interesting image segments from the image
    img_segments = extract_boxes_from_img(img, corners)

    # draw = ImageDraw.Draw(img)
    # for corner in corners:
    #     draw.rectangle(corner) 
    # img = np.asarray(img)
    visualize(img, show_img)
    return img_segments

def analog_yuv(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    yuv = np.zeros(img.shape)
    yuv[:, :, 0] =  0.299*r + 0.587*g + 0.114*b
    yuv[:, :, 1] = -0.169*r - 0.332*g + 0.500*b + 128
    yuv[:, :, 2] =  0.500*r - 0.419*g - 0.081*b + 128
    return yuv

def yCbCr_to_bin(img, meanY, meanCb, meanCr, stdY, stdCb, stdCr, factor):
    # yCbCr_img = img.convert('YCbCr')
    # yCbCr_img = np.array(yCbCr_img)
    yCbCr_img = analog_yuv(img)
    Y = yCbCr_img[:, :, 0]
    Cb_vals = yCbCr_img[:, :, 1]
    Cr_vals = yCbCr_img[:, :, 2]

    # min_Cb = meanCb - stdCb*factor
    # max_Cb = meanCb + stdCb*factor
    # min_Cr = meanCr - stdCr*factor
    # max_Cr = meanCr + stdCr*factor
    min_Cb, max_Cb = 76, 127
    min_Cr, max_Cr = 132, 173

    binImage = np.zeros(yCbCr_img.shape)
    Cb = np.zeros(Cb_vals.shape)      
    Cr = np.zeros(Cr_vals.shape)

    Cb[np.where( (Cb_vals > min_Cb) & (Cb_vals < max_Cb) )] = 1
    Cr[np.where( (Cr_vals > min_Cr) & (Cr_vals < max_Cr) )] = 1
    binImage = Cb*Cr
    return binImage

def gen_box(segment_idx):
    row, col = segment_idx
    min_x, max_x = min(col), max(col)
    min_y, max_y = min(row), max(row)
    center = ((min_x + max_x) // 2, (min_y + max_y) // 2)
    width = max(max_x - min_x, max_y - min_y)
    return (center, width)

def remove_area(img, P):
    segments, num_segments = label(img, background=0, return_num=True)
    for i in range(num_segments): 
        segment_idx = np.where(segments == i)
        if len(segment_idx[0]) < P: 
            # Remove this section altogether 
            img[segment_idx] = 0 
    return img

def extract_boxes_from_img(img, corners):
    regions_of_interest = []
    for corner in corners:
        top_left_x, top_left_y = corner[0]
        bottom_right_x, bottom_right_y = corner[1]
        # Adjust points to make we're within the image bounds
        top_left_x = max(top_left_x, 0)
        top_left_y = max(top_left_y, 0)
        bottom_right_x = min(bottom_right_x, img.shape[1])
        bottom_right_y = min(bottom_right_y, img.shape[0])
        # Add image segment to regions of interest
        img_segment = img[top_left_y:bottom_right_y,
                          top_left_x: bottom_right_x]
        regions_of_interest.append( (img_segment, corner) )
    return regions_of_interest
