# MACHINE LEARNING IN PYTHON - THE COURSERA COURSE I LEARNT

- Hello. This is going to be what I've learnt, mostly lab-related codes, on models of machine learning.
- This is also going to be my first time trying to use Git Hub, so I hope this will be interesting.
- Throughout my published lab coding problems and my comments for those, I will explain them all in this file (Though I think that's the purpose of this README.md file anyway...)
- Let's get to it! Also it's been months since the last time I've touch some of the lab problems and codes, so consider this as a revision of the things I've learnt, and what are some of the most important things I learnt from these.


<details>
<summary><strong>PART 1: LINEAR REGRESSION</strong></summary>

<details>
<summary>LAB #1: SIMPLE LINEAR REGRESSION</summary>


- I think t n- I think the name says it all. Given a dataset, you're trying to predict the target's value based on a certain feature. The feature should be the most correlated to the target as a result.
- Personally, this was the easiest lab problem compares to all of the remaining labs, and it make sense.

<details><summary>There were a lot of new terms I had to learn: </summary>

1. read_csv: This is the bridge that helps datasets from the Internet to be used in our local machine.

2. hist: Making a histogram that can then be graphed into a barchart thanks to matplotlib.

3. scatter: A scatter plot that show points on an xy-graph, but this time is between two feature.

4. reshape: On the process of inserting the data on the LinearRegression(), the <fit> by default requires the  value to be at least a 2D array (n-observations, n-features). But because on this particular dataset the X is 1D, the reshape allows the X to turn 1D to 2D. Not sure if I'll ever be able to use it in real life though. 
</details>

- Overall, the process of this lab problem is as follows:
Get the dataset --> Get it to our computer --> Preprocess the data --> Split into testing and training data --> Use the training data on the model --> Get the prediction data from the model ---> Compare it with the testing data to get our evaluation of the model.

- Everytime I come back to this lab problem, I can understand all the codes, but can't actually type them. I think that should be fine, since understanding what the code first should be prioritized.
- Everytime I come back to this lab problem, I can understand all the codes, but can't actually type them. I think that should be fine, since understanding what the code first should be prioritized.

Summary:

Input: Dataset from the Internet

Output: A linear regression model that predict the target's value from a most correlated feature.

</details>

<details>
<summary>LAB #2: MULTIPLE LINEAR REGRESSION</summary>

- Still simple concept. It's like having more features to predict a target, that's all.
- The hard part for me on this lab was trying to understand what the code was doing. Compare to LAB #1, this one felt like an actual boss when I first encounter it.

<details> <summary> This time I'll try to explain the code through each process, one by one: </summary>

1. Dataset from Internet to our computer: The exact same.

2. Preprocess the data <Part 1>: Now for the model, the data needs to be numerical (and relevant, not it right now), so any redundant or features that won't be useful for the model must be transformed, or removed. Categories can be transformed into numerical by using certain ways, but for this lab, we would remove it. The next step is also a a part of preprocessing process: Finding the correlation between features and targets by making scatter plots.

3. Preprocess the data <Part 2>: This one is a bit hard initally for me cause of <iloc>, but then it makes sense eventually. X is the entire two columns' data, and y is the 3rd column. The reason this is <Part 2> is because we have to <Standardize> the data, not just removing redundant or categorical features. The formula behind this... is annoying to learn. Basically we would turn the data so that it would have <mean = 0> and <Standard Deviation = 1>. And then X' = (X-m)/S, I think. Consider that a hint...

4. Split into testing and training data

5. Train the data on the model: The model is still called LinearRegression(), just having more features than the last lab.

6. Get the coefficients and intercepts of the model's training. This is the <Standard> version, meaning that the values here are made from a standardized dataset.

7. <Optional> Get the coefficients and intercepts of the actual, unstandardized data <Task 12>. This is probably for when we need to describe the relationship between the actual features and targets. The values on the <standard> version won't work because the data was not the same with the original version. The whole formula thing is like <y = mx' + n>, but this time substitute it with <x' = (x=-m)/S>.

8. Plot the model: Involves plotting 3 values, which requires a 3D regression plane. Also plotting each feature with the target, so 2 simple linear regression plot as well. </details>

Summary:

Input: Dataset
Output: More linear regression between 2 features and 1 target. Note that this can go in many ways, now that we learn about multiple linear regression.

EX: We might literally use 5 features to predict 3 targets.

</details>
</details>
</details>


<details>
<summary><strong>PART 2: REDUCE DIMENSION</strong></summary>

<details>
<summary>LAB #3: PCA (PRINCIPLE COMPONENTS ANALYSIS)</summary>

- Now we are getting to the interesting part. Reduce dimension algorithm.
- Imagine a dataset that is just a table. That is 2D. What if a 3D, 4D, 5D,...? How would we even begin to preprocess the data in the first place?
- This is where PCA comes in clutch. It reduces the dimensions of the data without losing much of the data's information, making the preprocessing stage much easier to deal with.

<details> <summary> Most of the instructions of how the code works are already in the code; here is a short outline of PCA: </summary>

- Given a dataset that's been graphed, PCA finds a line that captures the highest amount of variance (PC1). Points projected onto this line define the new coordinate.
- Given a dataset that's been graphed, PCA finds a line that captures the highest amount of variance (PC1). Points projected onto this line define the new coordinate.

- If choosing two components, a second principal component (PC2) is chosen orthogonal to PC1, and together they form a new coordinate system.

- The reduced dataset is the points' coordinates in the principal component axes. </details>

Summary:

Input: Dataset with too many dimensions
Output: Dataset with less dimensions.

</details>

<details>
<summary>LAB #4: t-SNE and UMAP</summary>

- These two dimensional reduction algorithms are competitors to PCA.
- The explanation of the code can be found in the code file.
s>
<summary><strong>Dimensionality reduction comparison: PCA vs t-SNE vs UMAP</strong></summary>

<div style="overflow-x:auto;"> 
<table style="border-collapse:collapse; width:100%; max-width:100%; font-family:Arial,Helvetica,sans-serif;">
	<thead>
		<tr>
			<th style="text-align:left; padding:12px; background:#f0f0f0; border:1px solid #ddd;">Feature</th>
			<th style="text-align:left; padding:12px; background:linear-gradient(90deg,#d81b60,#ff4081); color:#fff; border:1px solid #ddd;">PCA</th>
			<th style="text-align:left; padding:12px; background:linear-gradient(90deg,#00bcd4,#18ffff); color:#fff; border:1px solid #ddd;">t-SNE</th>
			<th style="text-align:left; padding:12px; background:linear-gradient(90deg,#8bc34a,#c5e1a5); color:#000; border:1px solid #ddd;">UMAP</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Type of method</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Linear method (uses linear transformations)</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Non-linear method (focuses on local structure)</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Non-linear method (focuses on both local and global structure)</td>
		</tr>
		<tr>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Focus</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Maximizing variance (global structure)</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Preserving local structure (neighborhoods)</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Preserving both local and global structure</td>
		</tr>
		<tr>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Preserves</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Variance (overall spread of the data)</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Local relationships (similarity between neighbors)</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Local and global relationships (overall shape and clusters)</td>
		</tr>
		<tr>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Output</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Linear transformation of the data into principal components</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">2D or 3D representation that reflects local similarities</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">2D or 3D representation with more global structure</td>
		</tr>
		<tr>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Scalability</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Highly scalable (works well with large datasets)</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Computationally expensive on large datasets</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Scalable, faster than t-SNE, works well with large datasets</td>
		</tr>
		<tr>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Speed</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Fast</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Slow, especially for large datasets</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Faster than t-SNE, more scalable</td>
		</tr>
		<tr>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Reproducibility</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Very stable and deterministic</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Results can vary with different runs</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">More stable than t-SNE, but less deterministic than PCA</td>
		</tr>
		<tr>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Interpretability</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Results are easy to interpret (principal components)</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Results are harder to interpret (abstract relationships)</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Results are more interpretable than t-SNE, but not as clear as PCA</td>
		</tr>
		<tr>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">Best for</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">When you want to preserve overall variance and reduce dimensionality linearly (e.g., for feature extraction, noise reduction)</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">When you want to explore local structure and identify clusters in the data</td>
			<td style="padding:10px; border:1px solid #eee; vertical-align:top;">When you want to preserve both local and global structure, especially in complex, large datasets</td>
		</tr>
		</tbody>
	</table>
	</div>

</details>

</details>

<details>
<summary><strong>PART 3: CLUSTERING</strong></summary>

<details>
<summary>LAB #5: K-MEANS CLUSTERING</summary>

- I kinda like this lab. It is visually intuitive and the plots make the behavior of the algorithm easy to understand.
- K-Means partitions data into k non-overlapping clusters. Each cluster has a centroid and the algorithm iteratively assigns points to the nearest centroid and recomputes centroids until convergence.
- The method assumes roughly spherical, similarly sized clusters and is sensitive to outliers.

Examples / notes:

1. Choose k (number of clusters) up front — there are heuristics to help with this (see below).
2. K-Means tries to minimize within-cluster variance (sum of squared distances to centroids).
3. It works well on well-separated clusters and scales to moderately large datasets.

Important caveats:

- Assumes clusters are roughly convex and similarly sized.
- Sensitive to outliers and initialization (use multiple restarts / k-means++ initialization).

- Heuristic techniques for choosing k and evaluating clustering quality:

	- Silhouette analysis

	- Elbow method

	- Davies-Bouldin index

</details>

</details>

<details>
<summary><strong>PART 4: CLASSIFICATION</strong></summary>

- This is related to supervised learning model. It uses fully trained model to predict labels in new data. It aims to predict data in the correct context when answering a specific question. Basically we have labels, new data goes to those labels by predicting which one it belongs to.

- As data is input into the model, the model adjusts the data to fit the algorithm and classifies it accordingly, defining the input and the predicted output. So it's like we have a framework of a model, and that frame can be shaped based on the data input that we have, creating newly shaped data predicted output.

- Several important applications of this model: Relation between features and targets, email filterings, speech-to-text,...
For instance:
- Churn prediction is when you use machine learning classification to predict whether a customer will discontinue a service. 
- Customer segmentation is when you use classification to predict the category to which a customer belongs.

- Binary classification is a model that only predict two possible results.
- Some common machine learning classification algorithms include Naive Bayes, Logistic Regression, Decision Trees, K-Nearest Neighbors, Support Vector Machines, and Neural Networks.
- Algorithms like Logistic Regression, KNN, and Decision Trees can learn how to distinguish multiple classes.
- Strategies for extending binary classifiers to handle multiple classes include one-versus-all classification and one-versus-one classification. (OvA is like a classifier differentiate their own data and the rest; OvO have each classifier compare the data to be either A or B.)
<details>
<summary>LAB #6: MULTICLASS CLASSIFICATION</summary>

- Even though this lab is suppose to be more focused on the model itself, the most important part of this lab that lasts in thee future is going to be: Preprocess the data.
- Specifically, standardize the numerical data and use one-hot encoding for categorical datas.
- One-hot encoding is mostly for features, but we can also do that to the targets if the labels are also categoricals. For instace, we don't need one-hot encoding on churn prediction, cause it is just 0 and 1.
- Anyway the process is still pretty much the same: Get the data -> Preprocess the data (either by standardize, remove irrelevant data, one-hot encoding,...) --> Split into x and y --> Turn it to training and testing data --> Train the model --> Evaluate the model (by getting the predicted result and compare it with the actual result)
- For this lab, there is an actual function that summarizes this entire process, I'll put up there as well as the original lab problem. Once again, focus on the process, and then the actual code syntax.
</details>

<details>
<summary> LAB #7: DECISION TREE </summary>

- The model kinda looks like a tree, and it's one of the type that are like, P(A) P(B) P(A u B), you know. 
- Anyway the overall process does not change much, other than we have a new approach to turn categorical label back to numerical: LabelEncoder(), instead of using OneHotEncoding. Honestly when I learn this the first time, had a hard time figuring out when to use either of these for specific data, and here's my answer: Ask AI if this data should be encoded by this or that:)
- There is something interesting about the model: We get to choose the max_depth of the model to figure out if we can optimize its accuracy. In the lab, we had to repeat the block of code to try max_depth =3. In the future, when we learn about the "pipeline", we don't have to repeat and do that thing anymore. Glad to look back and realize that certain codes can be optimized in the future.
- Also, kinda worry about the preprocess the data part. The fact that real life data is gonna be complex, hard to figure out the important data that needs to be kept, the ones that are irrelevant, and figure out the correlation between the features and the target,... Let's hope that by the time I sent all of my labs, I am gonna be more confident in making a ML model.
</details>

<details>
<summary> LAB #8: REGRESSION TREE </summary>

- This model is different from decision tree. For decision tree, it is for predicting discrete classes. For this model, it is predicting continuous model.
- In decision tree, the target is categorical. In regression tree, the target is a continuous value.
- When a decision tree is used to solve regression problem, it is called Regression Tree.
- Regression trees are created by recursively splitting the dataset into subsets to maximize information gained from data splitting.
- This model tries to pick features that minimized errors between actual values and predicted ones.
- Also the accuracy of the model is based on MSE and R^2 score. We want lower MSE and higher R^2 score for a good model.
</details>
</details>