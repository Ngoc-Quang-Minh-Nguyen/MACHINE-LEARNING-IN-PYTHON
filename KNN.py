import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()
df['custcat'].value_counts()
# The data set is mostly balanced between the different classes 
# and requires no special means of accounting for class bias.

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.show()
# Visualize the correlation map of the data set to determine
# how the different features are related to each other.

correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
print(correlation_values)
# This shows us that the features retire and gender have the least effect on custcat
# while ed and tenure have the most effect.

# Separate the features and targets
X = df.drop('custcat',axis=1)
y = df['custcat']

# Normalize data
X_norm = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

k = 6
#Train Model and Predict
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)


#In multilabel classification, accuracy classification score is a function that computes subset accuracy.
#This function is equal to the jaccard_score function.
#Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.
yhat = knn_model.predict(X_test)
print("Test set Accuracy: ", accuracy_score(y_test, yhat))

# Choosing the correct value of K
Ks = 500
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #Train Model and Predict
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1)
# The weak performance on the model can be due to multiple reasons. 
# 1. The KNN model relies entirely on the raw feature space at inference time.
# If the features do not provide clear boundaries between classes,
# KNN model cannot compensate through optimization or feature transformation. 
# 2. For a high number of weakly correlated features, the number of dimensions increases, 
# the distance between points tend to become more uniform, reducing the discriminative power of KNN.
# 3. The algorithm treats all features equally when computing distances. 
# Hence, weakly correalted features can introduce noise or irrelevant variations in the feature space 
# , making it harder for KNN to find meaningful neighbours.