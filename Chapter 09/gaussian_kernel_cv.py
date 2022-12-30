import cv2 as cv
import numpy as np
from os import path

IMAGE_DIR = r"images"
IMAGE_NAME = r"wiki_ind"
IMAGE_EXT = r".jpg"
IMAGE_PATH = path.join(IMAGE_DIR, IMAGE_NAME+IMAGE_EXT)
X_START = Y_START = -3
X_STOP = Y_STOP = 3

def read_image(path):
    image = cv.imread(path)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image_gray

def gaussian_fn_2d(x, y):
    # Exponent part of gaussain function mean=0, sd=1
    exp_part = np.exp(-(np.power(x, 2.) + np.power(y, 2.)) / 2.)
    return exp_part
def get_gaussian_kernel():
    kernel_shape = (X_STOP-X_START+1, Y_STOP-Y_START+1)
    gaussian_sample = np.zeros(shape=kernel_shape, dtype=float)
    ker_x = ker_y = 0
    for y_idx in range(Y_START, Y_STOP + 1):
        for x_idx in range(X_START, X_STOP+1):
            gaussian_sample[ker_x][ker_y] = gaussian_fn_2d(x_idx, y_idx)
            ker_y = ker_y + 1  # updating index
        ker_x = ker_x + 1  # Updating index
        ker_y = 0
    return gaussian_sample


# apply gaussain kernel
def apply_kernel_repeat(src_image, kern2d, repetition=1):
    filtered_img = src_image
    for idx in range(repetition):
        filtered_img = cv.filter2D(filtered_img,ddepth=-1,kernel=kern2d)
    return filtered_img


gaussain_ker = get_gaussian_kernel()
# Supressing scientific notation
np.set_printoptions(suppress=True)
print(gaussain_ker)
sum_kernel = np.sum(gaussain_ker)
print(sum_kernel)

div_part = 2. * np.pi
norm_gauss_kernel = (1./div_part) * gaussain_ker
#norm_kernel = (1./sum_kernel) * gaussain_ker
print("Normalized Kernel\n", norm_gauss_kernel)
print("Sum of Normalized Kernel:", np.sum(norm_gauss_kernel))

# Image reading
image_gray = read_image(IMAGE_PATH)
cv.imshow("Gray Image", image_gray)
gray_image_path = path.join(IMAGE_DIR, IMAGE_NAME + "_gr" + IMAGE_EXT)
cv.imwrite(gray_image_path, image_gray)

# Apply kernel
#image_filtered = apply_kernel_repeat(image, norm_kernel, repetition=1)
gauss_filtered_img = cv.filter2D(
                    image_gray, ddepth=-1,
                    kernel=norm_gauss_kernel)
cv.imshow("Smoothed Image", gauss_filtered_img)
mod_image_path = path.join(IMAGE_DIR, IMAGE_NAME + "_gauss_smooth" + IMAGE_EXT)
cv.imwrite(mod_image_path, gauss_filtered_img)

# Incorrect application of kernel without normalization
#image_filtered = apply_kernel_repeat(image, gaussain_ker)
#cv.imshow("Output Gauss Image", image_filtered)

cv.waitKey()
cv.destroyAllWindows()