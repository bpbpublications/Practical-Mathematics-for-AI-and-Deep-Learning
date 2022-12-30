import matplotlib.pyplot as plt
import cv2 as cv

IMG = r"C:\Users\skbm\OneDrive - Intel Corporation\Documents\DOCS\LearningAlgo" \
      r"\ML_RL_DL\writing_book\Computer Vision\images\ellipse.png"

ellipse = cv.imread(IMG)

# properties
print("Type:{}, SIze:{}", type(ellipse), ellipse.shape)

plt.imshow(ellipse)
plt.show()
