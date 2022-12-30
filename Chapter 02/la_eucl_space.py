# Projection
import matplotlib.pyplot as plt
import numpy as np

ax = plt.axes()

# Arrows
ax.arrow(0, 0, 10, 2, length_includes_head=True, head_width=0.5, head_length=0.5, ec='grey',
         overhang=1, alpha=.5)
ax.arrow(0, 0, 6, 6, length_includes_head=True, head_width=0.5, head_length=0.5, ec='grey',
         overhang=1, alpha=.5)
ax.arrow(0, 0, 6.86, 1.37, length_includes_head=True, head_width=0.5, head_length=0.5, ec='black',
         overhang=1, alpha=.5)
ax.arrow(0, 0, 10, 0, length_includes_head=True, head_width=0.5, head_length=0.5, ec='black',
         overhang=1, alpha=.5)

# Line
x_vals=[6,6.92];y_vals=[6,1.384]
ax.plot(x_vals, y_vals, linestyle='--',c='grey', alpha=1)
x_vals=[10,10];y_vals=[2,0]
ax.plot(x_vals, y_vals, linestyle='--',c='grey', alpha=1)

# Point
ax.plot(*np.array((6.86, 1.37)), marker="o")

# Plot text
ax.text(*np.array((10, 2)), r'$\mathbf{u}=(10,2)$')
ax.text(*np.array((6, 6)), r'$\mathbf{v}=(6,6)$')
ax.text(*np.array((1, .6)), r'$33.7\degree$')
ax.text(*np.array((2, .1)), r'$11.3\degree$')
ax.text(*np.array((5.7, 1.6)), r'$90\degree$')
ax.text(*np.array((9, .3)), r'$90\degree$')
ax.text(*np.array((6.86, 1)), r'$(6.92, 1.38)$')
ax.text(*np.array((2.6,0.7)), r'$\Vert\mathbf{v}\Vert\cos33.7\degree$',  rotation=11.3)
ax.text(*np.array((6.4,3.8)), r'$\Vert\mathbf{v}\Vert\sin33.7\degree$',  rotation=11.3)
ax.text(*np.array((2.5,3)), r'$\Vert\mathbf{v}\Vert=6\sqrt{2}\approx8.4$',  rotation=48)
ax.text(*np.array((6,0.2)), r'$\Vert\mathbf{u}\Vert\cos11.3\degree$')
ax.text(*np.array((10.1,0.8)), r'$\Vert\mathbf{u}\Vert\sin11.3\degree$')
ax.text(*np.array((7.8,1.8)), r'$\Vert\mathbf{u}\Vert=\sqrt{104}\approx10.2$',  rotation=45)
#ax.text(*l2, 'text rotated correctly', fontsize=12, rotation=angle, rotation_mode='anchor',
#              transform_rotates_text=True)

# Axis at (0,0)
ax.spines.left.set_position('zero')
ax.spines.bottom.set_position('zero')

# remove border lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Angle
# Arc
ax.annotate("",
            xy=(1, .2), xycoords='data',
            xytext=(.75, .75), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            color="0.5",
                            patchA=None,
                            shrinkA=0,shrinkB=0,
                            connectionstyle="arc3,rad=-.3",
                            ),
            )
ax.annotate("",
            xy=(1.5, .3), xycoords='data',
            xytext=(1.6, 0), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            color="0.5",
                            patchA=None,
                            shrinkA=0,shrinkB=0,
                            connectionstyle="arc3,rad=.3",
                            ),
            )
# 90 degree angle
ax.annotate("",
            xy=(6.8, 1.7), xycoords='data',
            xytext=(6.51, 1.33), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            color="0.5",
                            patchA=None,
                            shrinkA=0,shrinkB=0,
                            connectionstyle="angle,angleA=100,angleB=11.3,rad=0",
                            ),
            )
ax.annotate("",
            xy=(10, .3), xycoords='data',
            xytext=(9.7, 0), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            color="0.5",
                            patchA=None,
                            shrinkA=0,shrinkB=0,
                            connectionstyle="angle,angleA=90,angleB=0,rad=0",
                            ),
            )

# axis limit
plt.xlim(right=12)

plt.show()