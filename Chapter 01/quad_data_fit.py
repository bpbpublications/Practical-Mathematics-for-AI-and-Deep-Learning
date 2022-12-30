import matplotlib.pyplot as plt

in_x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
op_sq_noise = [26, 15, 10, 4, 2, 0, 1, 3, 8, 17, 23]
op_sq = [val**2 for val in in_x]

#plt.plot(in_x, op_sq_noise)
plt.scatter(in_x, op_sq_noise, c='grey')

op_p2 = [val*2 for val in in_x]
plt.plot(in_x, op_p2, ls='dotted', label='f=2x', c='grey')

op_pm4 = [val*-4 for val in in_x]
plt.plot(in_x, op_pm4, ls='--', label='f=-4x', c='grey')

op_p4 = [val*4 for val in in_x]
plt.plot(in_x, op_p4, ls='-.', label='f=4x', c='grey')

op_psq2 = [val**2 for val in in_x]
plt.plot(in_x, op_psq2, ls='-', label='g=x^2', c='grey')

plt.legend()
plt.show()
