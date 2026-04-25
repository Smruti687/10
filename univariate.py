import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
sns.histplot(tips['total_bill'], bins=20, kde=True)
plt.title('Histogram of Total Bill')

plt.subplot(3, 1, 2)
sns.boxplot(x=tips['total_bill'])
plt.title('Boxplot of Total Bill')

plt.subplot(3, 1, 3)
sns.kdeplot(tips['total_bill'], fill=True)
plt.title('Kernel Density Estimation of Total Bill')

plt.tight_layout()
plt.show()
