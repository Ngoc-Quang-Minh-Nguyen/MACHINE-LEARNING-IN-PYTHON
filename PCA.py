# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Generate bivariate normal distribution data. 
# Bivibrate means two features. Cov means the slope of the data, and how much the two features vary together.
# Mean means the center of the data.
np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 2]]
X = np.random.multivariate_normal(mean=mean, cov=cov, size=200)

# Scatter plot of the two features
plt.figure()
plt.scatter(X[:, 0], X[:, 1],  edgecolor='k', alpha=0.7)
plt.title("Scatter Plot of Bivariate Normal Distribution")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis('equal')
plt.grid(True)
plt.show()


# Apply PCA to the generated data. n_components=2 means we want to reduce the data to 2 principal components. 
# Original data already has 2 features. So we will see how PCA transforms the data. 
# If the original data had more than 2 features, PCA would reduce it to 2 features.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Print the principal components. The numbers mean the direction of the new axes in the original feature space.
# Basically, the numbers mean the slope of the new axes. The slope of the old data is given by the covariance matrix.
components = pca.components_
print(components)

# Print explained variance ratio. This tells us how much variance is explained by each principal component.
# The first principal component explains the most variance, the second explains the second most, and so on.
# Variance is a measure of how much the data varies. Higher variance means more information.
print(pca.explained_variance_ratio_)

# Project the original data onto the principal components. 
# Basically a perpendicular projection of the data points onto the new axes defined by the principal components.
# The PCA is now our new Ox and Oy axis. And we are projecting the original data onto these new axes.
projection_pc1 = np.dot(X, components[0]) # DATA IN THE ORIGINAL SPACE BUT GOT PC'ed. This is the projection onto PC1.
projection_pc2 = np.dot(X, components[1]) # This is the projection onto PC2.

x_pc1 = projection_pc1 * components[0][0] # DATA IN THE ORIGINAL SPACE BUT GOT PC'ed, This is the x coordinate of the projection onto PC1.
y_pc1 = projection_pc1 * components[0][1] # This is the y coordinate of the projection onto PC1.
x_pc2 = projection_pc2 * components[1][0] # Same for PC2.
y_pc2 = projection_pc2 * components[1][1] # Same for PC2.

# Plot original data
plt.figure()
plt.scatter(X[:, 0], X[:, 1], label='Original Data', ec='k', s=50, alpha=0.6)

# Plot the projections along PC1 and PC2
plt.scatter(x_pc1, y_pc1, c='r', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 1')
plt.scatter(x_pc2, y_pc2, c='b', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 2')
plt.title('Linearly Correlated Data Projected onto Principal Components', )
plt.xlabel('Feature 1',)
plt.ylabel('Feature 2',)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()


# Load the Iris dataset. This is a classic dataset for testing PCA.
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
print(target_names)
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA and reduce the dataset to 2 components. The Iris dataset has 4 features, so we reduce it to 2.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA-transformed data in 2D
plt.figure(figsize=(8,6))

# Define colors and plot settings. The colors correspond to the three species of Iris flowers in the dataset.
colors = ['navy', 'turquoise', 'darkorange']
lw = 1
# Plot each class with a different color.
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, s=50, ec='k',alpha=0.7, lw=lw,
                label=target_name)

plt.title('PCA 2-dimensional reduction of IRIS dataset',)
plt.xlabel("PC1",)
plt.ylabel("PC2",)
plt.legend(loc='best', shadow=False, scatterpoints=1,)
# plt.grid(True)
plt.show()
# This number shows how much variance is explained by the two principal components combined.
print(100*pca.explained_variance_ratio_.sum())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA. This time we keep 4 components, which is the same as the original number of features.
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio. This shows how much variance is explained by each of the 4 principal components.
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio for each component
plt.figure(figsize=(10,6))
plt.bar(x=range(1, len(explained_variance_ratio)+1), height=explained_variance_ratio, alpha=1, align='center', label='PC explained variance ratio' )
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Explained Variance by Principal Components')

# Plot cumulative explained variance. 
# The red dashed line shows how much variance is explained as we add more components.
# It should reach close to 1 (or 100%) as we include all components.
cumulative_variance = np.cumsum(explained_variance_ratio)
plt.step(range(1, 5), cumulative_variance, where='mid', linestyle='--', lw=3,color='red', label='Cumulative Explained Variance')
# Only display integer ticks on the x-axis. Interger ticks means only whole numbers. 
# If we don't do this, matplotlib will show decimal ticks like 1.0, 1.5, 2.0, etc. 
# Now it will only show 1, 2, 3, 4.
plt.xticks(range(1, 5))
plt.legend()
plt.grid(True)
plt.show()