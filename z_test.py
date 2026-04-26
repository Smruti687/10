import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# given data
sample_mean = 38
population_mean = 40
std_dev = 4
sample_size = 25
significance_level = 0.05

# calculate Z and p-value
z_value = (sample_mean - population_mean) / (std_dev / np.sqrt(sample_size))
p_value = 2 * (1 - norm.cdf(abs(z_value)))

# plot normal curve
x_values = np.linspace(-4, 4, 200)
y_values = norm.pdf(x_values)
plt.plot(x_values, y_values)

# shade critical region
critical_x = np.linspace(abs(z_value), 4, 100)
plt.fill_between(critical_x, norm.pdf(critical_x), alpha=0.5)

# draw Z lines
plt.axvline(abs(z_value), linestyle='--', label='Z value')
plt.axvline(-abs(z_value), linestyle='--')
plt.title('Z-test for Mean')
plt.legend()
plt.show()

# results
print("Z value =", z_value)
print("p value =", p_value)

if p_value < significance_level:
    print("Reject H0")
else:
    print("Fail to reject H0")
