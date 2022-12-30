"""
OpenCV provides two transformation functions, cv2.warpAffine and cv2.warpPerspective,
with which you can have all kinds of transformations.
cv2.warpAffine takes a 2x3 transformation matrix while cv2.warpPerspective
takes a 3x3 transformation matrix as input.
https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
https://opencv-tutorial.readthedocs.io/en/latest/trans/transform.html
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import argparse

IMG_PATH = r"C:\Users\skbm\OneDrive - Intel Corporation\Documents\DOCS\LearningAlgo" \
           r"\ML_RL_DL\writing_book\Computer Vision\images\letterE.png"


def rotate(image, size, centre, angle):
    rot_matrix = cv.getRotationMatrix2D(centre, angle, 1.0)
    print("rot_matrix", rot_matrix)
    rot_image = cv.warpAffine(image, rot_matrix, size)
    return rot_image


def translate(image, x, y, output_sz):
    print("Translating image with tx={0} & ty={1}".format(x,y))
    translated_mat = np.float32([[1,0,x],[0,1,y]])
    translated_img = cv.warpAffine(image, translated_mat, output_sz)
    return translated_img


def scale(image, centre, scale_ft, output_sz):
    scale_matrix = cv.getRotationMatrix2D(centre, 0, scale_ft)
    print("Scaling matrix:", scale_matrix)
    scaled_img = cv.warpAffine(image, scale_matrix, output_sz)
    return scaled_img


def scale_along_axis(image, x, y, output_sz):
    scale_matrix = np.float32([[x, 0, 0], [0, y, 0]])
    print("Scaling along axis matrix:", scale_matrix)
    scaled_img = cv.warpAffine(image, scale_matrix, output_sz)
    return scaled_img


def shear_along_axis(image, x, y, output_sz):
    shear_matrix = np.float32([[1, x, 0], [y, 1, 0]])
    print("Shearing along axis matrix:", shear_matrix)
    scaled_img = cv.warpAffine(image, shear_matrix, output_sz)
    return scaled_img


def main():
    ap = argparse.ArgumentParser()
    # Add arguments
    ap.add_argument("-i", "--image", default=IMG_PATH, help="input image")
    ap.add_argument("-t", "--trans", type=str,
                    choices=['translate', 'scale', 'scale_axis', 'shear_axis', 'rotate'],
                    default='shear_axis',
                    help="select type of affine transformation")
    ap.add_argument("-a", "--angle", type=float, default=90, help="rotation angle")
    ap.add_argument("-s", "--scale", type=float, default=1.5, help="scaling factor")
    ap.add_argument("-x", "--alongx", type=float, default=0.2, help="transformation value along x-axis")
    ap.add_argument("-y", "--alongy", type=float, default=0.3, help="transformation value along y-axis")
    # Parse the arguments
    args = ap.parse_args()

    # Reading image
    image = cv.imread(args.image)
    # rows = height, cols = width
    rows, cols = image.shape[:2]
    output_sz = (cols, rows)
    centre = cols//2, rows//2

    if args.trans == 'rotate':
        angle = args.angle
        print("Rotating with", angle)
        trans_img = rotate(image, size=output_sz, centre=centre, angle=angle)

    elif args.trans == 'translate':
        trans_x = args.alongx
        trans_y = args.alongy
        print("Translating image")
        trans_img = translate(image, trans_x, trans_y, output_sz)

    elif args.trans == 'scale_axis':
        scale_x = args.alongx
        scale_y = args.alongy
        print("scaling along axis of the image")
        trans_img = scale_along_axis(image, scale_x, scale_y, output_sz)

    elif args.trans == 'scale':
        scale_ratio = args.scale
        print("Scaling the image")
        trans_img = scale(image, centre, scale_ratio, output_sz)

    elif args.trans == 'shear_axis':
        shear_x = args.alongx
        shear_y = args.alongy
        print("shearing along axis of the image")
        trans_img = shear_along_axis(image, shear_x, shear_y, output_sz)

    else:
        print("Invalid option: Displaying input image")
        trans_img = image

    # Display transformed image
    #plt.axis('off')
    #plt.imshow(trans_img)
    #plt.show()

    cv.imshow("Transformed", trans_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
