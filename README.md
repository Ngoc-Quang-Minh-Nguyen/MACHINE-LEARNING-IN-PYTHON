# MACHINE LEARNING IN PYTHON - THE COURSERA COURSE I LEARNT
- Hello. This is going to be what I've learnt, mostly lab-related codes, on models of machine learning.
- This is also going to be my first time trying to use Git Hub, so I hope this will be interesting.
- Throughout my published lab coding problems and my comments for those, I will explain them all in this file (Though I think that's the purpose of this README.md file anyway...)
- Let's get to it! Also it's been months since the last time I've touch some of the lab problems and codes, so consider this as a revision of the things I've learnt, and what are some of the most important things I learnt from these.

## LAB #1: SIMPLE LINEAR REGRESSION
- I think the name says it all. Given a dataset, you're trying to predict the target's value based on a certain feature. The feature should be the most correlated to the target as a result.
- Personally, this was the easiest lab problem compares to all of the remaining labs, and it make sense.
- There were a lot of new terms I had to learn:
1. read_csv: This is the bridge that helps datasets from the Internet to be used in our local machine.
2. hist: Making a histogram that can then be graphed into a barchart thanks to matplotlib.
3. scatter: A scatter plot that show points on an xy-graph, but this time is between two feature.
4. reshape: On the process of inserting the data on the LinearRegression(), the <fit> by default requires the  value to be at least a 2D array (n-observations, n-features). But because on this particular dataset the X is 1D, the reshape allows the X to turn 1D to 2D. Not sure if I'll ever be able to use it in real life though.

- Overall, the process of this lab problem is as follows:
Get the dataset --> Get it to our computer --> Preprocess the data --> Split into testing and training data --> Use the training data on the model --> Get the prediction data from the model ---> Compare it with the testing data to get our evaluation of the model.

- Everytime I come back to this lab problem, I can understand all the codes, but can't actually type them. I think that should be fine, since understanding what the code first should be prioritized. 

Summary: Input: Dataset from the Internet
         Output: A linear regression model that predict the target's value from a most correlated feature. 
