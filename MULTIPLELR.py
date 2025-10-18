# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split

# Task 1: Load dataset from URL
# Input: URL to CSV file
# Expected Output: Pandas DataFrame containing fuel consumption and CO2 data
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)

# Task 2: Drop categorical and irrelevant columns
# Input: Raw DataFrame
# Expected Output: Cleaned DataFrame with only numeric predictors and target
df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE'], axis=1)

# Task 3: Inspect correlations between variables
# Input: Cleaned DataFrame
# Expected Output: Correlation matrix showing relationships between features and target
print(df.corr())

# Task 4: Drop redundant or highly correlated features
# Input: Correlation matrix and domain knowledge
# Expected Output: Reduced DataFrame with selected predictors
df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB'], axis=1)

# Task 5: Display sample data and updated correlations
# Input: Reduced DataFrame
# Expected Output: First few rows and new correlation matrix
print(df.head(9))
print(df.corr())

# Task 6: Visualize relationships between variables
# Input: Final DataFrame
# Expected Output: Scatter matrix plot with rotated axis labels
axes = pd.plotting.scatter_matrix(df, alpha=0.2, color = 'red')
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')
plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()

# Task 7: Separate predictors and target
# Input: Final DataFrame
# Expected Output: Numpy arrays X (predictors) and y (target)
X = df.iloc[:, [0, 1]].to_numpy()
y = df.iloc[:, [2]].to_numpy()

# Task 8: Standardize predictors
# Input: Raw X array
# Expected Output: Standardized X array with mean 0 and std 1
std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)
print(pd.DataFrame(X_std).describe().round(2))

# Task 9: Split data into training and test sets
# Input: Standardized X and y arrays
# Expected Output: X_train, X_test, y_train, y_test arrays
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# Task 10: Train linear regression model
# Input: Training data
# Expected Output: Fitted model with learned coefficients
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Task 11: Print standardized coefficients and intercept
# Input: Trained model
# Expected Output: Coefficients and intercept in standardized space
coef_ = regressor.coef_
intercept_ = regressor.intercept_
print('Coefficients: ', coef_)
print('Intercept: ', intercept_)

# Task 12: Convert coefficients to original scale
# Input: Standard scaler parameters and model coefficients
# Expected Output: Coefficients and intercept in original feature space
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)
coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)
print('Coefficients: ', coef_original)
print('Intercept: ', intercept_original)

# Task 13: Visualize regression plane in 3D
# Input: Test data and model predictions
# Expected Output: PREPARATION 3D plot showing regression plane and actual data points
X1 = X_test[:, 0]
X2 = X_test[:, 1]
x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100),
                               np.linspace(X2.min(), X2.max(), 100))
y_surf = intercept_ + coef_[0,0] * x1_surf + coef_[0,1] * x2_surf
y_pred = regressor.predict(X_test)
above_plane = y_test >= y_pred
below_plane = y_test < y_pred
above_plane = above_plane[:,0]
below_plane = below_plane[:,0]

#Creating the 3d graph
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane], label="Above Plane", s=70, alpha=.7, ec='k')
ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane], label="Below Plane", s=50, alpha=.3, ec='k')
ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21)
#Decoration
ax.view_init(elev=10)
ax.legend(fontsize='x-large', loc='upper center')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(None, zoom=0.75)
ax.set_xlabel('ENGINESIZE', fontsize='xx-large')
ax.set_ylabel('FUELCONSUMPTION_COMB_MPG', fontsize='xx-large')
ax.set_zlabel('CO2 Emissions', fontsize='xx-large')
ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large')
plt.tight_layout()
#Show it
plt.show()

# Task 14: Plot vertical slices (2D views) of regression fit
# Input: Training data and model coefficients
# Expected Output: Two 2D plots showing best-fit lines for each predictor
plt.scatter(X_train[:,0], y_train, color='blue')
plt.plot(X_train[:,0], coef_[0,0] * X_train[:,0] + intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(X_train[:,1], y_train, color='blue')
plt.plot(X_train[:,1], coef_[0,1] * X_train[:,1] + intercept_[0], '-r')
plt.xlabel("FUELCONSUMPTION_COMB_MPG")
plt.ylabel("Emission")
plt.show()