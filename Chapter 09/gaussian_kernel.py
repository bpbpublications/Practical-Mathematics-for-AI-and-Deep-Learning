import matplotlib.pyplot as plt
import numpy as np
#from skimage.io import imread, imshow
#from skimage.color import rgb2gray
from PIL.Image import open as popen
from scipy.signal import convolve2d

X_START = Y_START = -3
X_STOP = Y_STOP = 3
IMAGE_PATH = "test_wiki.png"
'''
Calculates values of gaussian function that has 
mean=0 and sigma=1
'''
def gaussian_fn_2d(x, y):
    # Exponent part of gaussain function mean=0, sd=1
    exp_part = np.exp(-(np.power(x, 2.) + np.power(y, 2.)) / 2.)
    return exp_part
def get_gaussian_kernel():
    kernel_shape = (X_STOP-X_START+1, Y_STOP-Y_START+1)
    gaussian_sample = np.zeros(shape=kernel_shape, dtype=float)
    ker_x = ker_y = 0
    for x_idx in range(X_START, X_STOP+1):
        for y_idx in range(Y_START, Y_STOP+1):
            gaussian_sample[ker_x][ker_y] = gaussian_fn_2d(x_idx, y_idx)
            ker_y = ker_y + 1  # updating index
        ker_x = ker_x + 1  # Updating index
        ker_y = 0
    return gaussian_sample


# Read image
def read_image_grey(path):
    #image = rgb2gray(imread(path))
    image = plt.imread(path)
    print("Image Dimen", image.shape)
    return image


def multi_convolver(image, kernel, iterations):
    for i in range(iterations):
        image = convolve2d(image, kernel, 'same', boundary = 'fill',
                           fillvalue = 0)
    return image


gaussain_ker = get_gaussian_kernel()
# Supressing scientific notation
np.set_printoptions(suppress=True)
print(gaussain_ker)
div_part = 2. * np.pi
print(div_part)
sum_kernel = np.sum(gaussain_ker)
print(sum_kernel)
norm_kernel = (1./div_part) * gaussain_ker
print(norm_kernel)

# apply gaussain kernel
image_gray = read_image_grey(IMAGE_PATH)
fig, ax = plt.subplots(dpi=2000)
#ax.imshow(image_gray, cmap='gray', vmin=0, vmax=255)
ax.imshow(image_gray)
ax.set_axis_off()

#convolved_img = multi_convolver(image_gray, norm_kernel, 3)
#plt.imshow(convolved_img, cmap='gray')

#fig, ax = plt.subplots(1,2)
#ax[0].imshow(image_gray, cmap='gray')
#ax[1].imshow(convolved_img, cmap='gray')

plt.show()
