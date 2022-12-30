import cv2 as cv
import numpy as np
from os.path import join
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

IMAGE_DIR = r"images"
IMAGE_NAME = r"wiki_ind_gauss_smooth"
IMAGE_EXT = r".jpg"
IMAGE_PATH = join(IMAGE_DIR, IMAGE_NAME+IMAGE_EXT)


def read_image(path):
    image = cv.imread(path)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image_gray


def apply_kernel(src_image, kern):
    filtered_img = cv.filter2D(src_image,ddepth=-1,kernel=kern)
    return filtered_img


def apply_threshold(image):
    # Global Thresholding
    global_th_img = image > threshold_otsu(image)
    global_th_name = join(IMAGE_DIR, IMAGE_NAME + "_glth" + IMAGE_EXT)
    cv.imwrite(global_th_name, global_th_img)
    # Local Thresholding
    window_size = 15
    thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.05)
    thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=0.05)
    niblack_bin_img = image > thresh_niblack
    sauvola_bin_img = image > thresh_sauvola
    nib_th_name = join(IMAGE_DIR, IMAGE_NAME + "_nibth" + IMAGE_EXT)
    cv.imwrite(nib_th_name, niblack_bin_img)
    sau_th_name = join(IMAGE_DIR, IMAGE_NAME + "_sauth" + IMAGE_EXT)
    cv.imwrite(sau_th_name, sauvola_bin_img)


def apply_kernel_save(src_image, kernx, kerny, suffix):
    filtered_img_x = cv.filter2D(src_image,ddepth=-1,kernel=kernx)
    filtered_img_y = cv.filter2D(src_image, ddepth=-1, kernel=kerny)
    cv.imshow("Sobel Filtered Image x" + suffix, filtered_img_x)
    cv.imshow("Sobel Filtered Image y" + suffix, filtered_img_y)
    img_name_x = join(IMAGE_DIR, IMAGE_NAME + "_sobx" + suffix + IMAGE_EXT)
    img_name_y = join(IMAGE_DIR, IMAGE_NAME + "_soby" + suffix + IMAGE_EXT)
    img_name_xy = join(IMAGE_DIR, IMAGE_NAME + "_sobxy" + suffix + IMAGE_EXT)
    cv.imwrite(img_name_x, filtered_img_x)
    cv.imwrite(img_name_y, filtered_img_y)
    filtered_img_xy = filtered_img_x + filtered_img_y
    cv.imshow("Sobel Filtered Image xy" + suffix, filtered_img_xy)
    cv.imwrite(img_name_xy, filtered_img_xy)
    # Pass the image to thresholding
    apply_threshold(img_name_x)


def get_sobel_kernel(dim):
    sobelx_sep = cv.getDerivKernels(1, 0, dim, normalize=True)
    sobelx = np.outer(sobelx_sep[0], sobelx_sep[1])
    sobely_sep = cv.getDerivKernels(0, 1, dim, normalize=True)
    sobely = np.outer(sobely_sep[0], sobely_sep[1])
    print("sobelx0:\n{0}, sobelx1:\n{1}".format(sobelx_sep[0], sobelx_sep[1]))
    print("sobely0:\n{0}, sobely1:\n{1}".format(sobely_sep[0], sobely_sep[1]))
    return sobelx, sobely


# Image reading
image = read_image(IMAGE_PATH)
cv.imshow("Input Image", image)

kern_size = [1,3,5,7]
for kern_dim in kern_size:
    kernelx, kernely = get_sobel_kernel(kern_dim)
    apply_kernel_save(image, kernelx, kernely, str(kern_dim))


cv.waitKey()
cv.destroyAllWindows()