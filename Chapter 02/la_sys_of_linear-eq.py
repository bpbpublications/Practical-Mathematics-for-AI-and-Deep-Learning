# Infinite number of solutions
import numpy as np
import matplotlib.pyplot as plt

# create the figure
fig = plt.figure()

# add axes
ax = fig.add_subplot(projection='3d')
axis_b_start = 300; axis_b_end = 500
axis_p_start = 400; axis_p_end = 600

b1, p1 = np.meshgrid(range(axis_b_start, axis_b_end), range(axis_p_start, axis_p_end))
m1 = (1500 - b1 - p1)
# plot the plane
ax.plot_surface(b1, p1, m1, color='cyan', alpha=0.4)

b2 = b1; p2 = p1
m2 = (4400 - 3*b2 - 4*p2) / 2
# plot the plane
ax.plot_surface(b2, p2, m2, color='red', alpha=0.4)

b3 = b1; p3 = p1
m3 = (2000 - 2*b3 - 2*p3) / 2
# plot the plane
ax.plot_surface(b3, p3, m3, color='orange', alpha=0.4)

# Display labels
ax.set_xlabel('Bowling Alleys', fontsize='x-large')
ax.set_ylabel('Play Stations', fontsize='x-large')
ax.set_zlabel('Movie Tickets', fontsize='x-large')
#ax.set_title("3 planes do not intersect at single point", fontsize='x-large')

plt.show()