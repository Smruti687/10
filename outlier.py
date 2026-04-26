import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor

iris = load_iris().data
y_pred = LocalOutlierFactor(n_neighbors=20).fit_predict(iris)

plt.scatter(iris[:,0], iris[:,1], c=y_pred)
plt.colorbar()
plt.title('Outlier Detection with Local Outlier Factor (LOF)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()
