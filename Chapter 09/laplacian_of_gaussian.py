import cv2 as cv
import numpy as np
from os.path import join

IMAGE_DIR = r"images"
IMAGE_NAME = r"wiki_ind"
IMAGE_EXT = r".jpg"
IMAGE_PATH = join(IMAGE_DIR, IMAGE_NAME+IMAGE_EXT)
X_START = Y_START = -3
X_STOP = Y_STOP = 3
KERNEL_CENTRE_X = KERNEL_CENTRE_Y = X_STOP


def read_image(path):
    image = cv.imread(path)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image_gray

def laplacian_of_gaussian_2d(x, y):
    # Exponent part of gaussain function mean=0, sd=1
    power_part = (np.power(x, 2.) + np.power(y, 2.)) / 2.
    exp_part = np.exp(-power_part)
    prod_part = (-1./np.pi) * (1. - power_part)
    return prod_part * exp_part

def get_log_kernel():
    kernel_shape = (X_STOP-X_START+1, Y_STOP-Y_START+1)
    log_sample = np.zeros(shape=kernel_shape, dtype=float)
    ker_x = ker_y = 0
    for x_idx in range(X_START, X_STOP+1):
        for y_idx in range(Y_START, Y_STOP+1):
            log_sample[ker_x][ker_y] = laplacian_of_gaussian_2d(x_idx, y_idx)
            ker_y = ker_y + 1  # updating index
        ker_x = ker_x + 1  # Updating index
        ker_y = 0
    return log_sample

def apply_kernel_repeat(src_image, kern2d, repetition=1):
    filtered_img = src_image
    for idx in range(repetition):
        filtered_img = cv.filter2D(filtered_img,ddepth=-1,kernel=kern2d)
    return filtered_img


log_ker = get_log_kernel()
# Supressing scientific notation
np.set_printoptions(suppress=True)
print(log_ker)

sum_kernel = np.sum(log_ker)
print("Kernel Sum", sum_kernel)
# Add sum to centre of kernel
log_ker[KERNEL_CENTRE_X][KERNEL_CENTRE_Y] = log_ker[KERNEL_CENTRE_X][KERNEL_CENTRE_Y] - sum_kernel
sum_kernel = np.sum(log_ker)
print("Kernel Sum (after adjustment)", sum_kernel)

# Multiply kernel
log_ker = log_ker * 10.0
print("After Scaling:", log_ker)

# Image reading
image = read_image(IMAGE_PATH)
cv.imshow("Input Image", image)
# Apply kernel
image_filtered = apply_kernel_repeat(image, log_ker, repetition=1)
cv.imshow("Filtered Image", image_filtered)
filt_img_name = join(IMAGE_DIR, IMAGE_NAME+"_log_filt"+IMAGE_EXT)
cv.imwrite(filt_img_name, image_filtered)
# Add to original image
image_add = image - image_filtered
cv.imshow("Image LOG Add", image_add)
img_add_name = join(IMAGE_DIR, IMAGE_NAME+"_log_add"+IMAGE_EXT)
cv.imwrite(img_add_name, image_add)

cv.waitKey()
cv.destroyAllWindows()