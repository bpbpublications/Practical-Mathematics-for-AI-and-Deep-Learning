# Transformation Eigen Values Vectors
import matplotlib.pyplot as plt
import numpy as np

ax = plt.axes()

# Transformation matrix
trans_mat = np.array([[3,1],[1,3]])

# Eigen vectors of transformation matrix
eigen_vects = np.array([[3,3],[2,2],[3,-3],[2,-2]])
eigen_vects_st_pt = np.array([[-5,-4],[2,1],[0,-2],[-5,4]])

# Non eigen vectors
non_eigen_vects = np.array([[2,-1],[1,4],[-1,-4],[-2,-3]])
non_eigen_vects_st_pt = np.array([[-3,5],[-2,-5],[2,5],[5,-1]])

# Points to be plotted
for x_index in range(-5,6):
    for y_index in range(-5,6):
        #ax.plot(*np.array((x_index, y_index)), marker="o", c='grey')
        # Multiplying vector by transformation matrix
        x_coor = trans_mat[0][0]*x_index + trans_mat[0][1]*y_index
        y_coor = trans_mat[1][0]*x_index + trans_mat[1][1]*y_index
        ax.plot(*np.array((x_coor, y_coor)), marker="o", c='grey', alpha=0.3)

# Transformation on Eigen vectors
for row_index in range(eigen_vects.shape[0]):
    # transformation on vector length
    x_coor = trans_mat[0][0]*eigen_vects[row_index][0] + trans_mat[0][1]*eigen_vects[row_index][1]
    y_coor = trans_mat[1][0]*eigen_vects[row_index][0] + trans_mat[1][1]*eigen_vects[row_index][1]
    # transformation on starting point of vector
    st_pt_x = trans_mat[0][0]*eigen_vects_st_pt[row_index][0] + trans_mat[0][1]*eigen_vects_st_pt[row_index][1]
    st_pt_y = trans_mat[1][0]*eigen_vects_st_pt[row_index][0] + trans_mat[1][1]*eigen_vects_st_pt[row_index][1]
    ax.arrow(st_pt_x, st_pt_y, x_coor, y_coor,
             length_includes_head=True, head_width=0.5,
             head_length=0.5, ec='black', fc='black', overhang=0, alpha=.8)

# Non Eigen Vectors
for row_index in range(non_eigen_vects.shape[0]):
    # transformation on vector length
    x_coor = trans_mat[0][0]*non_eigen_vects[row_index][0] + trans_mat[0][1]*non_eigen_vects[row_index][1]
    y_coor = trans_mat[1][0]*non_eigen_vects[row_index][0] + trans_mat[1][1]*non_eigen_vects[row_index][1]
    # transformation on starting point of vector
    st_pt_x = trans_mat[0][0]*non_eigen_vects_st_pt[row_index][0] + trans_mat[0][1]*non_eigen_vects_st_pt[row_index][1]
    st_pt_y = trans_mat[1][0]*non_eigen_vects_st_pt[row_index][0] + trans_mat[1][1]*non_eigen_vects_st_pt[row_index][1]
    ax.arrow(st_pt_x, st_pt_y, x_coor, y_coor,
             length_includes_head=True, head_width=0.5, ls='-.',
             head_length=0.5, ec='black', fc='black', overhang=0, alpha=.5)

# Test
ax.text(*np.array((-9.62,6.68)), r'$\mathbf{e_2}=(4, -4)$')
ax.text(*np.array((-4.24,13.2)), r'$\mathbf{v_1}=(5,-1)$')
ax.text(*np.array((10.25,15.26)), r'$\mathbf{v_2}=(-7, -13)$')
ax.text(*np.array((10.7,8.2)), r'$\mathbf{e_1}=(8,8)$')
ax.text(*np.array((9.18,-4.73)), r'$\mathbf{v_4}=(-9,-11)$')
ax.text(*np.array((0.5,-7.75)), r'$\mathbf{e_2}=(6,-6)$')
ax.text(*np.array((-8.56,-13.32)), r'$\mathbf{v_3}=(7,13)$')
ax.text(*np.array((-16.19,-14.78)), r'$\mathbf{e_1}=(12,12)$')

# Axis at (0,0)
ax.spines.left.set_position('zero')
ax.spines.bottom.set_position('zero')

# remove border lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()