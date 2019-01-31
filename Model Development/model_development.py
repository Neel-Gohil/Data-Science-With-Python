# Neel Gohil
# Python for Data Science - Model Development

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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