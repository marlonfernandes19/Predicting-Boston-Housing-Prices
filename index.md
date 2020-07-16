## Predicting Boston Housing Prices


The objective/goal of the project is to obtain an optimal model based on a statistical data to estimate the best price for a customer to buy a house in Boston, Massachusetts. 

### Boston Housing Dataset 
The dataset used for this project is included in scikit-learn library (sklearn.datasets.load_boston) which the data collected from homes in suburbs of Boston, Massachusetts. The expense of a house varies according to various factors like average number of rooms per dwelling, full-value property-tax rate per $10,000,  pupil-teacher ratio, crime rate, etc.

### Algorithm : 
#### Linear regression algorithm
Linear regression algorithm is one of the fundamental supervised machine-learning algorithms due to its relative simplicity and well-known properties.Regression is a method of modelling a target value based on independent predictors. It is mostly used for forecasting and finding out cause and effect relationship between variables.

Simple linear regression is a type of regression analysis where the number of independent variables is one and there is a linear relationship between the independent(x) and dependent(y) variable.
```
y = mx + c
```

To know more about Linear regression algorithm, [Click here](https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a)


## Now let's start 

### 1. Import libraries 
First we must import the following Python libraries :

* numpy - for arrays and matrix processing with the help of a large collection of high-level mathematical functions
* pandas - for data analysis
* matplotlib.pyplot - for data visualization
* seaborn - for making statistical graphics in Python and data visualization library based on matplotlib

```python3
import numpy as np 
import pandas as pd 
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
%matplotlib inline

```
### 2. Load the dataset
Now we are going to import the Boston Housing dataset and store it in a variable called boston_data.
```python3

from sklearn.datasets import load_boston
boston_data = load_boston()
```
First let's check for its keys.
```python3
boston_data.keys()
```
*Output : dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])*

```python3
boston_data.feature_names
```
*Out  : array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')*

Now let's view the description of the dataset.
```python3
boston_data.DESCR
```
![data_DESCR](https://github.com/marlonfernandes19/Predicting-Boston-Housing-Prices/blob/master/res/boston_data.DESCR.png)

```python3
boston_data.data.shape
```
*Out  : (506, 13)*
```python3
boston_data.target.shape
```
*Out  : (506,)*

```python3
bos_df = pd.DataFrame(boston_data.data)
bos_df.columns = boston_data.feature_names
bos_df.head()
```
![df_col](https://github.com/marlonfernandes19/Predicting-Boston-Housing-Prices/blob/master/res/df_column.png)


```python3
bos_df.describe()
```
![df_des](https://github.com/marlonfernandes19/Predicting-Boston-Housing-Prices/blob/master/res/df_describe.png)

```python3
x = bos_df
y = pd.DataFrame(boston_data.target)
```

```python3
from sklearn import linear_model
from sklearn.model_selection import train_test_split
```
Initialize the linear regression model
```python3
l_reg = linear_model.LinearRegression()
```

```python3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=57)
l_reg.fit(x_train, y_train)
```

*Out  : LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)*


coef
```python3
coeff_df = pd.DataFrame(l_reg.coef_.flatten() ,x.columns, columns=['Coefficient'])
coeff_df
```
![coeff](https://github.com/marlonfernandes19/Predicting-Boston-Housing-Prices/blob/master/res/df_describe.png)

Display the Intercept
```python3
print(l_reg.intercept_)
```
*Out  : [44.32583168]*


predictions
```python3
predictions = l_reg.predict(x_test)
```

print the actual price of houses from the testing data set
```python3
y_test[0]
```
![act_price]()

Let us see the Price for
```python3
predictions[2]
```
*Out  : array([17.12720781])*

Let see the graph
```python3
plt.scatter(y_test, predictions, edgecolor='Darkred', color='orange')
plt.xlabel('Prices')
plt.ylabel('Predicted prices')
plt.title('Prices vs Predicted prices')

```
![graph1]()

```
sns.distplot((y_test - predictions), bins = 50, hist_kws=dict(edgecolor="black", linewidth=1),color='Darkred')
```
![dist_1]()

# Now check model performance/accuracy using,
# mean squared error which tells you how close a regression line is to a set of points.
```python3
from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test, predictions))
```
*Out  : 23.026198484642887*

```python3
l_reg.score(x_test,y_test)
```
*Out  : 0.7368039496310854*

```python3
from sklearn import ensemble
gbr = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=3, min_samples_split=3, learning_rate=0.3, loss='ls')
gbr.fit(x_train,np.ravel(y_train,order='C'))
```

*Out :*
*GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.3, loss='ls', max_depth=3,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=3,
                          min_weight_fraction_leaf=0.0, n_estimators=500,
                          n_iter_no_change=None, presort='deprecated',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)*
```python3
new_predictions = gbr.predict(x_test)
```

```python3
plt.scatter(y_test, new_predictions, edgecolor='Darkred', color='orange')
plt.xlabel('Prices')
plt.ylabel('Predicted prices')
plt.title('Prices vs Predicted prices')
```
![graph2]()
```python3
new_predictions[2]
```
*Out : 14.853728385701617*
```python3
gbr.score(x_test,y_test)
```
*Out :  0.9066266412990818*


