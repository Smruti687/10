import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mean, sd = 150, 20

x = np.linspace(50, 250, 100)
y = norm.pdf(x, mean, sd)

plt.plot(x, y)

# highlight regions
plt.fill_between(x, y, where=(x>=140)&(x<=160), alpha=0.5, label='140 to 160')
plt.fill_between(x, y, where=(x>170), alpha=0.5, label='>170')
plt.fill_between(x, y, where=(x<120), alpha=0.5, label='<120')
plt.xlabel('Weight (grams)')
plt.ylabel('Probability Density')
plt.title('Normal Distribution of Apple Weights')
plt.legend()
plt.show()

# probabilities
p1 = norm.cdf(160, mean, sd) - norm.cdf(140, mean, sd)
p2 = 1 - norm.cdf(170, mean, sd)
p3 = norm.cdf(120, mean, sd)

print(f'P(140 < X < 160): {p1:.4f}')
print(f'P(X > 170): {p2:.4f}')
print(f'P(X < 120): {p3:.4f}')
