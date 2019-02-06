#Neel Gohil
#Python for Data Science - Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

path='https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'

df = pd.read_csv(path)

'''Observing the data set and looking for correlation'''
#print(df.head())

#print(df.dtypes)

#print(df.corr())

#print(df[['bore','stroke','compression-ratio','horsepower']].corr())

'''Engine size as potential predictor variable of price'''

sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
# Plot shows a positive linear relationship between engine size and price
#plt.show()

# Double check the plot using numerical values'''
print(df[['engine-size', 'price']].corr())
#relationship of 0.87

'''Look at highway-mpg and price of car to observe correlation'''
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
#shows negative linear relationship
#plt.show()

# Double check the plot using numerical values
print(df[['highway-mpg', 'price']].corr())
# relationship of -0.70

'''Observation of weak linear relationship'''
sns.regplot(x="peak-rpm", y="price", data=df)
#plt.show()
print(df[['peak-rpm','price']].corr())
# relationship of -0.10

'''Relationship of categorical values'''
#look at the relationship between "body-style" and "price"

print(sns.boxplot(x="body-style", y="price", data=df))
#plt.show()
#Significant Overlap between the prices so this comparison would not be good for price prediction


print(sns.boxplot(x="engine-location", y="price", data=df))
#plt.show()
# Price in this plot is distinct making this comparison a better predictor of price

'''Descriptive Statistical Analysis'''

print(df.describe())
#computes basic statistics

print(df.describe(include= 'object'))
#include type object

'''Value Counts'''

print(df['drive-wheels'].value_counts())
# prints out characteristics of the variable

print(df['drive-wheels'].value_counts().to_frame())
#convert to data frame


#change column name from drive-wheels to value-counts and print all the characteristics
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
print(drive_wheels_counts)

#Same as above methods but for engine location and set name for index
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts)

'''Grouping'''

print(df['drive-wheels'].unique())
#prints all the different categories

#select the columns 'drive-wheels','body-style' and 'price' , then assign it to the variable "df_group_one".
df_group_one=df[['drive-wheels','body-style','price']]

#calculate the average price for each of the different categories of data.
df_group_one=df_group_one.groupby(['drive-wheels'],as_index= False).mean()
print(df_group_one)

#groups the dataframe by the unique combinations 'drive-wheels' and 'body-style'. We can store the results in the variable 'grouped_test1'
df_gptest=df[['drive-wheels','body-style','price']]
grouped_test1=df_gptest.groupby(['drive-wheels','body-style'],as_index= False).mean()
print(grouped_test1)

#Find average price of each car based on body-style
df_make = df['make'].unique()
df_make=df[['make', 'body-style', 'price']]
grouped_test2 = df_make.groupby(['make', 'body-style'], as_index = False).mean()
print(grouped_test2)

grouped_pivot=grouped_test2.pivot(index='make', columns='body-style')
print(grouped_pivot)

grouped_pivot=grouped_pivot.fillna(0) #fill missing values with 0
print(grouped_pivot)

'''Pearson Correlation'''

#calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'.

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#Since the p-value is < 0.001, the correlation between wheel-base and price is statistically significant,


#calculate the Pearson Correlation Coefficient and P-value of 'horsepower' and 'price'
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


#calculate the Pearson Correlation Coefficient and P-value of 'length' and 'price'.
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#calculate the Pearson Correlation Coefficient and P-value of 'width' and 'price'
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )

#calculate the Pearson Correlation Coefficient and P-value of 'curb-weight' and 'price'
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#calculate the Pearson Correlation Coefficient and P-value of 'engine-size' and 'price'
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#calculate the Pearson Correlation Coefficient and P-value of 'bore' and 'price'
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )

#calculate the Pearson Correlation Coefficient and P-value of 'city-mpg' and 'price'
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#calculate the Pearson Correlation Coefficient and P-value of 'highway-mpg' and 'price'
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )


'''ANOVA Testing'''

#Let's see if different types 'drive-wheels' impact 'price', we group the data

grouped_test3= df_gptest[['drive-wheels','price']].groupby(['drive-wheels'])
print(grouped_test3.head(2))

#obtain the values of the method group using the method "get_group"
print(grouped_test3.get_group('4wd')['price'])

#use the function 'f_oneway' in the module 'stats' to obtain the F-test score and P-value.

f_val, p_val = stats.f_oneway(grouped_test3.get_group('fwd')['price'], grouped_test3.get_group('rwd')['price'],
                              grouped_test3.get_group('4wd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)

#Gives us great results stating price is correlated with drive wheels


'''We now have a better idea of what our data looks like and which variables are important to take into account when predicting the car price. We have narrowed it down to the following variables:

Continuous numerical variables:

Length
Width
Curb-weight
Engine-size
Horsepower
City-mpg
Highway-mpg
Wheel-base
Bore
Categorical variables:

Drive-wheels'''
