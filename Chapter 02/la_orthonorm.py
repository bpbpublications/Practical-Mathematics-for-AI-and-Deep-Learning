# Orthogonalization
import matplotlib.pyplot as plt
import numpy as np

ax = plt.axes()

# Arrows
ax.arrow(0, 0, 5, 2, length_includes_head=True, head_width=0.5, head_length=0.5, ec='black',
         overhang=1, alpha=1)
ax.arrow(0, 0, 3, 5, length_includes_head=True, head_width=0.5, head_length=0.5, ec='black',
         overhang=1, alpha=1)
ax.arrow(0, 0, 4.3, 1.7, length_includes_head=True, head_width=0.5, head_length=0.5, ec='black',
         overhang=1, alpha=.5)
ax.arrow(3, 5, -4.3, -1.7, length_includes_head=True, head_width=0.5, head_length=0.5, ec='black',
         overhang=1, alpha=.5)
ax.arrow(0, 0, -1.3, 3.3, length_includes_head=True, head_width=0.5, head_length=0.5, ec='black',
         overhang=1, alpha=.5)

# Line
x_vals=[3,4.3];y_vals=[5,1.7]
ax.plot(x_vals, y_vals, linestyle='--',c='black', alpha=.5)

# Point
ax.plot(*np.array((4.3, 1.7)), marker="o")

# Plot text
ax.text(*np.array((5.1, 2)), r'$\mathbf{v_1}=(5,2)$')
ax.text(*np.array((3, 5.1)), r'$\mathbf{v_2}=(3,5)$')
ax.text(*np.array((.05, .4)), r'$90\degree$')
ax.text(*np.array((3.7, 1.9)), r'$90\degree$')

proj = r'$\frac{\mathbf{v_1}\cdot\mathbf{v_2}}{\mathbf{v_1}\cdot\mathbf{v_1}}\mathbf{v_1}}\mathbf{v_1}$'
ax.text(*np.array((1.8,0.2)), proj,  rotation=20, fontsize=18)
minus_proj = r'$\minus\frac{\mathbf{v_1}\cdot\mathbf{v_2}}{\mathbf{v_1}\cdot\mathbf{v_1}}\mathbf{v_1}}\mathbf{v_1}$'
ax.text(*np.array((0.36,4.2)), minus_proj,  rotation=20, fontsize=18)
ax.text(*np.array((1,2.5)), r'$\mathbf{v_2}$',  rotation=59)
ortho = r'$\mathbf{v_2}\minus\frac{\mathbf{v_1}\cdot\mathbf{v_2}}{\mathbf{v_1}\cdot\mathbf{v_1}}\mathbf{v_1}}\mathbf{v_1}$'
ax.text(*np.array((-1.7,1)), ortho, fontsize=14)
#ax.text(*l2, 'text rotated correctly', fontsize=12, rotation=angle, rotation_mode='anchor',
#              transform_rotates_text=True)

# Axis at (0,0)
ax.spines.left.set_position('zero')
ax.spines.bottom.set_position('zero')

# remove border lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 90 degree angle
ax.annotate("",
            xy=(-0.09, .23), xycoords='data',
            xytext=(.17, .07), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            color="0.5",
                            patchA=None,
                            shrinkA=0,shrinkB=0,
                            connectionstyle="angle,angleA=111.8,angleB=21.8,rad=0",
                            ),
            )
ax.annotate("",
            xy=(4.18, 1.97), xycoords='data',
            xytext=(4.09, 1.65), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            color="0.5",
                            patchA=None,
                            shrinkA=0,shrinkB=0,
                            connectionstyle="angle,angleA=111.8,angleB=21.8,rad=0",
                            ),
            )

plt.show()