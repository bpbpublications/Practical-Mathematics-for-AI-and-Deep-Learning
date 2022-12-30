"""
OpenCV provides two transformation functions, cv2.warpAffine and cv2.warpPerspective,
with which you can have all kinds of transformations.
cv2.warpAffine takes a 2x3 transformation matrix while cv2.warpPerspective
takes a 3x3 transformation matrix as input.
https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
https://opencv-tutorial.readthedocs.io/en/latest/trans/transform.html
"""

import cv2 as cv
import numpy as np
import argparse
from os import path

IMAGE_DIR = r"images"
IMAGE_NAME = r"letterE"
IMAGE_EXT = r".jpg"
IMAGE_PATH = path.join(IMAGE_DIR, IMAGE_NAME+IMAGE_EXT)


def read_image(path):
    image = cv.imread(path)
    # convert image to grayscale
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image_gray


def rotate(image, angle_degree=20):
    (rows, cols) = image.shape[:2]
    radians = np.deg2rad(angle_degree)
    cos_val = np.cos(radians); sin_val = np.sin(radians)
    rotate_matrix = np.float32(
                [[cos_val,-1*sin_val,0],[sin_val,cos_val,0]])
    np.set_printoptions(suppress=True)
    print("Rotation Matrix\n", rotate_matrix)
    # applies transformation matrix to every pixel coordinate
    rotate_image = cv.warpAffine(image, rotate_matrix, (cols, rows))
    return rotate_image


def translate(image, trans_x, trans_y):
    (rows, cols) = image.shape[:2]
    translated_mat = np.float32([[1,0,trans_x],[0,1,trans_y]])
    print("Translation Matrix\n", translated_mat)
    translated_img = cv.warpAffine(image, translated_mat, (cols, rows))
    return translated_img


def scale(image, scale_x=0.3, scale_y=0.8):
    (rows, cols) = image.shape[:2]
    scaled_matrix = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])
    print("Scaling Matrix\n", scaled_matrix)
    scaled_img = cv.warpAffine(image, scaled_matrix, (cols, rows))
    return scaled_img


def shear(image, shear_x, shear_y):
    (rows, cols) = image.shape[:2]
    shear_matrix = np.float32([[1, shear_y, 0], [shear_x, 1, 0]])
    print("Shear Matrix\n", shear_matrix)
    scaled_img = cv.warpAffine(image, shear_matrix, (cols, rows))
    return scaled_img


def main():
    ap = argparse.ArgumentParser()
    # Add arguments
    ap.add_argument("-tx", "--trans_x", type=float, default=150, help="translation factor along x")
    ap.add_argument("-ty", "--trans_y", type=float, default=100, help="translation factor along y")
    ap.add_argument("-a", "--angle", type=float, default=20, help="rotation angle")
    ap.add_argument("-scx", "--scale_x", type=float, default=0.3, help="scaling factor along x")
    ap.add_argument("-scy", "--scale_y", type=float, default=0.8, help="scaling factor along y")
    ap.add_argument("-sx", "--shear_x", type=float, default=0.1, help="shear along x-axis")
    ap.add_argument("-sy", "--shear_y", type=float, default=0.5, help="shear along y-axis")
    # Parse the arguments
    args = ap.parse_args()

    # Reading image
    image = read_image(IMAGE_PATH)

    # Rotation
    angle = args.angle
    print("Rotating with", angle)
    rotate_img = rotate(image, angle)
    cv.imshow("Rotated", rotate_img)
    rot_img_name = path.join(IMAGE_DIR, IMAGE_NAME+"_rot"+IMAGE_EXT)
    cv.imwrite(rot_img_name, rotate_img)

    # Scaling
    sc_x = args.scale_x; sc_y = args.scale_y
    print("Scaling the image")
    scaled_img = scale(image, sc_x, sc_y)
    cv.imshow("Scaled", scaled_img)
    scaled_img_name = path.join(IMAGE_DIR, IMAGE_NAME+"_scal"+IMAGE_EXT)
    cv.imwrite(scaled_img_name, scaled_img)

    # Translation
    trans_x = args.trans_x; trans_y = args.trans_y
    print("Translating image")
    trans_img = translate(image, trans_x, trans_y)
    cv.imshow("Translated", trans_img)
    tran_img_name = path.join(IMAGE_DIR, IMAGE_NAME + "_tran" + IMAGE_EXT)
    cv.imwrite(tran_img_name, trans_img)

    # Shear
    shear_x = args.shear_x; shear_y = args.shear_y
    print("shearing along axis of the image")
    shear_img = shear(image, shear_x, shear_y)
    cv.imshow("SHear", shear_img)
    shear_img_name = path.join(IMAGE_DIR, IMAGE_NAME + "_shear" + IMAGE_EXT)
    cv.imwrite(shear_img_name, shear_img)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
