import cv2 as cv
import numpy as np
from os.path import join

IMAGE_DIR = r"images"
IMAGE_NAME = r"wiki_ind_gauss_smooth"
IMAGE_EXT = r".jpg"
IMAGE_PATH = join(IMAGE_DIR, IMAGE_NAME+IMAGE_EXT)


def read_image(path):
    image = cv.imread(path)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image_gray


def apply_kernel_repeat(src_image, kern2d, repetition=1):
    filtered_img = src_image
    for idx in range(repetition):
        filtered_img = cv.filter2D(filtered_img,ddepth=-1,kernel=kern2d)
    return filtered_img


#laplacian_kern = np.array([[0,1,0],[1,-4,1],[0,1,0]])
laplacian_kern_diag = np.array([[1,1,1],[1,-8,1],[1,1,1]])

# Supressing scientific notation
np.set_printoptions(suppress=True)
print(laplacian_kern_diag)

# Image reading
image = read_image(IMAGE_PATH)
cv.imshow("Input Image", image)
# Apply kernel
#image_filtered = apply_kernel_repeat(image, laplacian_kern)
#cv.imshow("Filtered no diagonal Image", image_filtered)

image_filtered_diag = apply_kernel_repeat(image, laplacian_kern_diag)
cv.imshow("Filtered diagonal Image", image_filtered_diag)
lap_img_name = join(IMAGE_DIR, IMAGE_NAME+"_lap"+IMAGE_EXT)
cv.imwrite(lap_img_name, image_filtered_diag)

# Adding filtered image
#final_img = image - image_filtered
#cv.imshow("Final Image(no diag)", final_img)

final_img_diag = image - image_filtered_diag
cv.imshow("Final Image(with diag)", final_img_diag)
lap_img_add_name = join(IMAGE_DIR, IMAGE_NAME+"_lap_add"+IMAGE_EXT)
cv.imwrite(lap_img_add_name, final_img_diag)

cv.waitKey()
cv.destroyAllWindows()