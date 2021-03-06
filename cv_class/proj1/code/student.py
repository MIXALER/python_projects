# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale


def my_imfilter(image, filter):
    """
    Your function should meet the requirements laid out on the project webpage.
    Apply a filter to an image. Return the filtered image.
    Inputs:
    - image -> numpy nd-array of dim (m, n, c)
    - filter -> numpy nd-array of odd dim (k, l)
    Returns
    - filtered_image -> numpy nd-array of dim (m, n, c)
    Errors if:
    - filter has any even dimension -> raise an Exception with a suitable error message.
    """
    #####################################################################################################
    #                                            Your Code                                              #
    #####################################################################################################
    filtered_image = np.zeros_like(image)
    x_pad = int((filter.shape[1] - 1) / 2)
    y_pad = int((filter.shape[0] - 1) / 2)
    if len(image.shape) == 2:
        image = np.pad(image, ((y_pad, y_pad), (x_pad, x_pad)), 'constant', constant_values=((0, 0), (0, 0)))
        for i in range(filtered_image.shape[0]):
            for j in range(filtered_image.shape[1]):
                part = image[i:i + filter.shape[0], j:j + filter.shape[1]]
                filtered_image[i, j] = (part * filter).sum()

    else:
        assert len(image.shape) == 3, "dims of input image error"
        image = np.pad(image, ((y_pad, y_pad), (x_pad, x_pad), (0, 0)), 'constant')

        for c in range(3):
            for i in range(filtered_image.shape[0]):
                for j in range(filtered_image.shape[1]):
                    part = image[i:i + filter.shape[0], j:j + filter.shape[1], c]
                    filtered_image[i, j, c] = (part * filter).sum()

    # assert any(filtered_image != None)
    #####################################################################################################
    #                                               End                                                 #
    #####################################################################################################

    return filtered_image


def normalize(img):
    ''' Function to normalize an input array to 0-1 '''
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min)


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    #####################################################################################################
    #                                            Your Code                                              #
    #####################################################################################################
    # Your code here:
    s, k = cutoff_frequency, cutoff_frequency * 2
    probs = np.asarray([exp(-z * z / (2 * s * s)) / sqrt(2 * pi * s * s) for z in range(-k, k + 1)], dtype=np.float32)
    kernel = np.outer(probs, probs)
    low_frequencies = my_imfilter(image1, kernel)  # Replace witih your implementation

    high_frequencies = image2 - my_imfilter(image2, kernel)  # Replace with your implementation

    hybrid_image = normalize(low_frequencies + high_frequencies)

    #####################################################################################################
    #                                               End                                                 #
    #####################################################################################################

    return low_frequencies, high_frequencies, hybrid_image


def vis_hybrid_image(hybrid_image):
    """
    Visualize a hybrid image by progressively downsampling the image and
    concatenating all of the images together.
    """
    scales = 5
    scale_factor = [0.5, 0.5, 1]
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales + 1):
        # add padding
        output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                            dtype=np.float32)))
        # downsample image
        cur_image = rescale(cur_image, scale_factor, mode='reflect')
        # pad the top to append to the output
        pad = np.ones((original_height - cur_image.shape[0], cur_image.shape[1],
                       num_colors), dtype=np.float32)
        tmp = np.vstack((pad, cur_image))
        output = np.hstack((output, tmp))
    return output


def load_image(path):
    return img_as_float32(io.imread(path))


def save_image(path, im):
    return io.imsave(path, img_as_ubyte(im.copy()))
