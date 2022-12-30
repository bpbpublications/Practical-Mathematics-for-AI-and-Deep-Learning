import matplotlib.pyplot as plt
import numpy as np

X_COOR = np.arange(1,16,1)
Y_COOR = np.array([5,5,5,5,4,3,2,1,1,1,1,4,4,4,4])

# Axis settings
plt.yticks(np.arange(0,6,1))
plt.xticks(np.arange(0,16,1))
plt.xlabel("Pixel Position")
plt.ylabel("Pixel Intensity")

plt.plot(X_COOR, Y_COOR, ls='--', marker='X')

plt.show()