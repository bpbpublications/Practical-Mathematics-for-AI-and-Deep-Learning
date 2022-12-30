# Hyperplane & Halfspaces
import matplotlib.pyplot as plt
import numpy as np

ax = plt.axes()

# Line
x_vals=[-6,9];y_vals=[6,-4]
ax.plot(x_vals, y_vals,c='black')

# Arrows
ax.arrow(1.5, 1, 2, 3, length_includes_head=True, head_width=0.5, head_length=0.5, ec='black',
         overhang=1, alpha=1)

# Plot text
#ax.text(*np.array((5.4, -1.3)), r'$2x+3y=6$')
#ax.text(*np.array((0.12, 2.6)), r'$2x+3y>6$')
#ax.text(*np.array((-4, 2)), r'$2x+3y<6$')

# Axis at (0,0)
ax.spines.left.set_position('zero')
ax.spines.bottom.set_position('zero')

# remove border lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
