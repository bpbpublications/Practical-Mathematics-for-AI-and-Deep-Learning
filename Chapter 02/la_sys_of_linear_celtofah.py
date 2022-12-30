import matplotlib.pyplot as plt
import numpy as np

ax = plt.axes()

x_vals=[-45,110];y_vals=[-49,230]
ax.plot(x_vals, y_vals,c='black')

# Line
x_vals=[100,100];y_vals=[0,212]
ax.plot(x_vals, y_vals, linestyle='--',c='black', alpha=.5)

# Line
x_vals=[0,100];y_vals=[212,212]
ax.plot(x_vals, y_vals, linestyle='--',c='black', alpha=.5)

# Point
ax.plot(*np.array((100, 212)), marker="o", c='grey')
ax.plot(*np.array((0, 32)), marker="o", c='grey')

# Plot text
ax.text(*np.array((100, 200)), r'$(100,212)$')
ax.text(*np.array((2, 20)), r'$32$')

# Display labels
ax.set_xlabel('Celsius C')
ax.set_ylabel('Fahrenheit F')

# Axis at (0,0)
ax.spines.left.set_position('zero')
ax.spines.bottom.set_position('zero')

# remove border lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
