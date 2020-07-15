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
![data1](https://github.com/marlonfernandes19/Predicting-Boston-Housing-Prices/blob/master/res/boston_data1.png)

```python3
boston_data.feature_names
```

Now let's view the description of the dataset.
```python3
boston_data.DESCR
```
![data_DESCR](https://github.com/marlonfernandes19/Predicting-Boston-Housing-Prices/blob/master/res/boston_data.DESCR.png)

```python3
boston_data.data.shape
```
```python3
boston_data.target.shape
```
![Target_shape](https://github.com/marlonfernandes19/Predicting-Boston-Housing-Prices/blob/master/res/target_shape.png)

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
```
Out : LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
```

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
```
Out : [44.32583168]
```

predictions
```python3
predictions = l_reg.predict(x_test)
```

print the actual price of houses from the testing data set
```python3
y_test[0]
```
You can use the [editor on GitHub](https://github.com/marlonfernandes19/marlonfernandes19.github.io/edit/master/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```
