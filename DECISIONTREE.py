#Library
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics



import warnings
warnings.filterwarnings('ignore')

#Insert data
path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)
my_data

#Check how the data looks like.
my_data.info()
# There are 4 lables that are labeled categorical. Gotta encode them. 
# For this, we can make use of LabelEncoder from the Scikit-Learn library.
label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex'])
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol'])
my_data
#Here's the result: Sex: M--> 1; F--> 0
#                   BP: High-->0 ; Low--> 1; Normal--> 2
#                   Cholesterol: High--> 0 ; Normal --> 1
#You can also check if there are any missing values in the dataset.
my_data.isnull().sum()
#There is none, so good.

# To evaluate the correlation of the target variable with the input features, 
# it will be convenient to map the different drugs to a numerical value. 
# Execute the following cell to achieve the same.
custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)

#Now we can find the inputs that are most correlated to the target.EX: Na_to_K is 0.589120, the highest.
print(my_data.drop('Drug',axis=1).corr()['Drug_num'])

# We can visually how much target's value we have for each type of drug.
category_counts = my_data['Drug'].value_counts()

# Plot the count plot
plt.bar(category_counts.index, category_counts.values, color='blue')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)  # Rotate labels for better readability if needed
plt.show()

#Prepare X and y
y = my_data['Drug']
X = my_data.drop(['Drug','Drug_num'], axis=1)

# Prepare traning and testing data
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)

# Prepare model with max depth = 4, put data in model, then use the model to predict new data. 
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)
tree_predictions = drugTree.predict(X_testset)

# Check how accurate it is with the actual testing data.
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))
plot_tree(drugTree)
plt.show()

# Same thing but with max depth = 3
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 3)
drugTree.fit(X_trainset,y_trainset)
tree_predictions = drugTree.predict(X_testset)
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))