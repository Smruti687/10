from scipy.stats import poisson
import matplotlib.pyplot as plt
import numpy as np

mean = 12
x = np.arange(0, 30)
y = poisson.pmf(x, mean)

plt.bar(x, y, alpha=0.4)

# highlight using fill_between
plt.fill_between(x, y, where=(x==5), alpha=0.6, label='x = 5')
plt.fill_between(x, y, where=(x<=12), alpha=0.4, label='x ≤ 12')
plt.fill_between(x, y, where=(x>=15), alpha=0.4, label='x ≥ 15')
plt.fill_between(x, y, where=(x>=10)&(x<=15), alpha=0.4, label='10 ≤ x ≤ 15')
plt.xlabel('Number of Accidents')
plt.ylabel('Probability')
plt.title('Poisson Distribution of Accidents in a Manufacturing Plant')
plt.legend()
plt.show()

print(f"Probability of observing exactly 5 accidents: {poisson.pmf(5, mean)}")
print(f"Probability of observing not more than 12 accidents: {poisson.cdf(12, mean)}")
print(f"Probability of observing at least 15 accidents: {1 - poisson.cdf(14, mean)}")
print(f"Probability of observing between 10 and 15 accidents: {poisson.cdf(15, mean) - poisson.cdf(9, mean)}")
