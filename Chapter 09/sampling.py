import numpy as np
import matplotlib.pyplot as plot

# Get x values of the sine wave
x_vals = np.arange(0, 3.21, 0.01);

# Amplitude of the sine wave is sine of a variable like time
amplitude_vals = np.sin(x_vals)

# Noise
uniform_noise = np.random.uniform(0,0.1,np.size(x_vals))

# Add noise to amplitude
amp_noise_vals = amplitude_vals + uniform_noise

ax = plot.axes()

# Plot a sine wave using time and amplitude obtained for the sine wave
#ax.plot(x_vals, amplitude_vals, c='black')
#ax.plot(x_vals, amp_noise_vals, c='grey')

# Point
for index in range(0,len(x_vals),10):
    ax.plot(*np.array((x_vals[index], amp_noise_vals[index])), marker="o", c='black')


plot.axis('off')
plot.show()