import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

ellipse = mpatches.Ellipse((.5,.5), 1, .5)

fig, ax = plt.subplots()

ax.imshow(gradient, cmap=plt.get_cmap('Greys'))
ax.add_patch(ellipse)

ax.set_axis_off()
plt.show()
