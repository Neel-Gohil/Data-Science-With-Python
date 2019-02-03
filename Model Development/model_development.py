# Neel Gohil
# Python for Data Science - Model Development

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns

# Import Data
path = 'https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'
df = pd.read_csv(path)
df.head()


''' Linear Regression and Multiple Linear Regression '''

#Create Linear Regression Object
lm = LinearRegression()
lm

#QUESTION: How can highway mpg predict car price?

X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)
#output prediction
Yhat=lm.predict(X)
Yhat[0:5]

#EQUATION: Y = mx+b
#value of intercept a

lm.intercept_

#slope (m value)

lm.coef_

# Plugging in the actual values we get:
# price = - 821.73 x highway-mpg + 38423.31

#QUESTION 2: How can engine size predict car price?
lm1 = LinearRegression()
lm1

A = df[['engine-size']]
B = df[['price']]

#lm1.coef_
#lm1.intercept_

#Equation: Price = 166.86 x engine-size - 7963.34

''' Multi-linear Regression '''

#From Data Analysis section, it was found that horsepower, curb-weight, engine size, and highway effected car price
# This model here uses those predictor variables

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

#fit linear model with predictor variables and target will be price
lm.fit(Z, df['price'])

lm.intercept_
lm.coef_

# EQUATION: Price = 52.65851272 x horsepower + 4.69878948 x curb-weight + 81.95906216 x engine-size + 33.58258185 x highway-mpg -15678.742628061467

''' Model Evaluation using Visualization '''

#visualize Horsepower as potential predictor variable of price

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
#Plot describes higher mpg correlates to lower overall car price


''' Residual Plot '''

#used to check for variance to see whether linear or non-linear model is needed

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()
#plot shows that the points are not randomly spread out making the linear regression model a better fit for the data

''' Multiple Linear Regression Visualization '''

Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))

#Plot 2 separate plots: 1 for price and other for predictor variables
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
# It shows some overlapping but could be improved


''' Polynomial Regression and Pipelines '''
#Try fitting Polynomial Model
def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)
    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


print("done")

x = df['highway-mpg']
y = df['price']
print("done")

# Here we use a polynomial of the 3rd order (cubic)
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

PlotPolly(p,x,y, 'highway-mpg')
#Plot has more overlaps

#Create 11th order polynomial model for giggles
f1 = np.polyfit(x,y,11)
p1 = np.poly1d(f1)
print(p1)

PlotPolly(p1, x, y, 'Highway-mpg')

#Polynomial Transformation

pr=PolynomialFeatures(degree=2)
pr

Z_pr=pr.fit_transform(Z)
print(Z_pr.shape)
# Before transformation, there were 201 samples with 4 features while after transformation has 15 features

''' Pipeline '''

Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

pipe=Pipeline(Input)
print(pipe)

pipe.fit(Z,y)

ypipe=pipe.predict(Z)
print(ypipe[0:4])

'''Measures for In-Sample Evaluation'''

#Model 1: Simple Linear Regression

#highway_mpg_fit

print(lm.fit(X, Y))
# Find the R^2
print(lm.score(X, Y))
#49.659% of the variation of the price is explained by this simple linear model

#Calculate Mean Square Error which is to see the difference between actual y value versus the predicted y value

Yhat=lm.predict(X)
print(Yhat[0:4])

#mean_squared_error(Y_true, Y_predict)
print(mean_squared_error(df['price'], Yhat))


#Model 2 multiple linear regression

# fit the model
lm.fit(Z, df['price'])
# Find the R^2
print(lm.score(Z, df['price']))
#80.896 % of the variation of price is explained by this multiple linear regression

#calculate Means Squared Error

Y_predict_multifit = lm.predict(Z)
mean_squared_error(df['price'], Y_predict_multifit)

#Model 3: Polynomial Fit
r_squared = r2_score(y, p(x))
print(r_squared)
#67.419 % of the variation of price is explained by this polynomial fit

print(mean_squared_error(df['price'], p(x)))

''' Prediction and Decision Making '''

new_input=np.arange(1,100,1).reshape(-1,1)

#fit the model
lm.fit(X, Y)
lm

#create prediciton
yhat=lm.predict(new_input)
yhat[0:5]

#plot data
plt.plot(new_input,yhat)
plt.show()

