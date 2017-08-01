# Boston-Housing-Market-Analysis

In this project, I will evaluate the performance and predictive power of a model that has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts. A model trained on this data that is seen as a good fit could then be used to make certain predictions about a home — in particular, its monetary value. This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis. The dataset for this project originates from the UCI Machine Learning Repository. The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. This dataset has the following fields: 

1. RM: average number of rooms per dwelling (independent variable)
2. LSTAT: percentage of population considered lower status (independent variable)
3. PTRATIO: pupil-teacher ratio by town (independent variable)
4. MEDV: median value of owner-occupied homes (dependent variable)

 I will use this Data-set to train a price-predicting model. I will first explore the data and determine a descriptive statistics about the data set. I will then split the data to training, corss-validation, and testing sets and use different perfomance graphs to find the best model.

For the purposes of this project, the following preprocessing steps have been made to the dataset:

16 data points have an 'MEDV' value of 50.0. These data points likely contain missing or censored values and have been removed.
1 data point has an 'RM' value of 8.78. This data point can be considered an outlier and has been removed.The features 'RM', 'LSTAT', 'PTRATIO', and 'MEDV' are essential. The remaining non-relevant features have been excluded.
The feature 'MEDV' has been multiplicatively scaled to account for 35 years of market inflation.
