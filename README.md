# MACHINE LEARNING IN PYTHON - THE COURSERA COURSE I LEARNT

- Hello. This is going to be what I've learnt, mostly lab-related codes, on models of machine learning.
- This is also going to be my first time trying to use Git Hub, so I hope this will be interesting.
- Throughout my published lab coding problems and my comments for those, I will explain them all in this file (Though I think that's the purpose of this README.md file anyway...)
- Let's get to it! Also it's been months since the last time I've touch some of the lab problems and codes, so consider this as a revision of the things I've learnt, and what are some of the most important things I learnt from these.


<details>
<summary>LAB #1: SIMPLE LINEAR REGRESSION</summary>


- I think the name says it all. Given a dataset, you're trying to predict the target's value based on a certain feature. The feature should be the most correlated to the target as a result.
- Personally, this was the easiest lab problem compares to all of the remaining labs, and it make sense.

<details> <summary>
- There were a lot of new terms I had to learn: </summary>

1. read_csv: This is the bridge that helps datasets from the Internet to be used in our local machine.

2. hist: Making a histogram that can then be graphed into a barchart thanks to matplotlib.

3. scatter: A scatter plot that show points on an xy-graph, but this time is between two feature.

4. reshape: On the process of inserting the data on the LinearRegression(), the <fit> by default requires the  value to be at least a 2D array (n-observations, n-features). But because on this particular dataset the X is 1D, the reshape allows the X to turn 1D to 2D. Not sure if I'll ever be able to use it in real life though.
</details>

- Overall, the process of this lab problem is as follows:
Get the dataset --> Get it to our computer --> Preprocess the data --> Split into testing and training data --> Use the training data on the model --> Get the prediction data from the model ---> Compare it with the testing data to get our evaluation of the model.

- Everytime I come back to this lab problem, I can understand all the codes, but can't actually type them. I think that should be fine, since understanding what the code first should be prioritized. 

Summary:

Input: Dataset from the Internet

Output: A linear regression model that predict the target's value from a most correlated feature.

</details>

<details>
<summary>LAB #2: MULTIPLE LINEAR REGRESSION</summary>


- Still simple concept. It's like having more features to predict a target, that's all.
- The hard part for me on this lab was trying to understand what the code was doing. Compare to LAB #1, this one felt like an actual boss when I first encounter it.

<details> 
<summary> This time I'll try to explain the code through each process, one by one: </summary>

1. Dataset from Internet to our computer: The exact same.

2. Preprocess the data <Part 1>: Now for the model, the data needs to be numerical (and relevant, not it right now), so any redundant or features that won't be useful for the model must be transformed, or removed. Categories can be transformed into numerical by using certain ways, but for this lab, we would remove it. The next step is also a a part of preprocessing process: Finding the correlation between features and targets by making scatter plots.

3. Preprocess the data <Part 2>: This one is a bit hard initally for me cause of <iloc>, but then it makes sense eventually. X is the entire two columns' data, and y is the 3rd column. The reason this is <Part 2> is because we have to <Standardize> the data, not just removing redundant or categorical features. The formula behind this... is annoying to learn. Basically we would turn the data so that it would have <mean = 0> and <Standard Deviation = 1>. And then X' = (X-m)/S, I think. Consider that a hint...

4. Split into testing and training data

5. Train the data on the model: The model is still called LinearRegression(), just having more features than the last lab.

6. Get the coefficients and intercepts of the model's training. This is the <Standard> version, meaning that the values here are made from a standardized dataset.

7. <Optional> Get the coefficients and intercepts of the actual, unstandardized data <Task 12>. This is probably for when we need to describe the relationship between the actual features and targets. The values on the <standard> version won't work because the data was not the same with the original version. The whole formula thing is like <y = mx' + n>, but this time substitute it with <x' = (x=-m)/S>.

8. Plot the model: Involves plotting 3 values, which requires a 3D regression plane. Also plotting each feature with the target, so 2 simple linear regression plot as well.
</details>

- This one took FOREVER for me to get through. Not just the coding part (I could just copy from the lab), but because I had a hard time understanding the code. Also it was so long, 150+ lines of codes.

Summary:

Input: Dataset
Output: More linear regression between 2 features and 1 target. Note that this can go in many ways, now that we learn about multiple linear regression.

EX: We might literally use 5 features to predict 3 targets.

</details>


<details>
<summary> LAB #3: PCA (PRINCIPLE COMPONENTS ANALYSIS) </summary>


- Now we are getting to the interesting part. Reduce dimension algorithm.
- Imagine a dataset that is just a table. That is 2D. What if a 3D, 4D, 5D,...? How would we even begin to preprocess the data in the first place?
- This is where PCA comes in clutch. It reduces the dimensions of the data without losing much of the data's information, making the preprocessing stage much easier to deal with.

<details> <summary>Most of the instructions of how the code works is already in the code, so for this outline, I will try to explain how PCA works: </summary>

- Given a dataset, let's say that it's already been graphed, PCA will try to find a certain line that can capture the HIGHEST AMOUNT OF VARIANCE. This line is called the First Principal Component (PC1), and it represents the new axis that the points should be projected onto. We can visualize what the line looks like by observing the locations of the projected points (like the red 'X' markers on the graph).

- If we choose a PCA dimension of 2, we need a second line to be perfectly perpendicular (orthogonal) to the first line. This second line is the Second Principal Component (PC2).

- Together, the PC1 and PC2 axes form a new coordinate system that replaces the old axis (Feature 1 and Feature 2). The final, reduced dataset consists of the points' new coordinates on these PC axes.
</details>

- I hope that should be good to understand for later uses. Be honest, we're probably just gonna come back to this lab if we ever need to use this model again, and just copy the parts that we need... Or DO WE???
- There might be a chance that we will only use a single line of code to apply the entire PCA model...we'll get to that point eventually. 
- Anyway that is all for this lab. I'll start to explain what the code does in the actuall code file, and this outline is just going to explain the process or theory behind each model. 

Summary:

Input: Dataset with too many dimensions
Output: Dataset with less dimensions.

</details>