#Neel Gohil
#Python for Data Science - Data Wrangling

import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot

'''Retrieving Dataset and adding Headers'''

filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(filename, names = headers)

#print("Done")
#print(df.head())

# Identify and Handle Missing Values '''
df.replace("?", np.nan, inplace = True)
#print(df.head(5))

# Evaluate for Missing Data
missing_data = df.isnull()
#print(missing_data.head(5))

'''Count missing values in each column
False = No missing value
True = missing value'''

# for column in missing_data.columns.values.tolist():
#     print(column)
#     print (missing_data[column].value_counts())
#     print("")

# Calculate the mean of the column
avg_1 = df["normalized-losses"].astype("float").mean(axis = 0)

# Replace "NaN" by mean value in "normalized-losses" column
df["normalized-losses"].replace(np.nan, avg_1, inplace = True)

# Calculate the mean of the column
avg_2=df['bore'].astype('float').mean(axis=0)

# Replace "NaN" by mean value
df['bore'].replace(np.nan, avg_2, inplace= True)

# Calculate the mean of the column
avg_3 = df['stroke'].astype('float').mean(axis=0)

#Replace "NaN" by mean value
df['stroke'].replace(np.nan, avg_3, inplace= True)

#Calculate the mean value for the  'horsepower' column
avg_4=df['horsepower'].astype('float').mean(axis=0)

# Replace "NaN" by mean value
df['horsepower'].replace(np.nan, avg_4, inplace= True)

# Calculate the mean value for 'peak-rpm' column
avg_5=df['peak-rpm'].astype('float').mean(axis=0)

# Replace NaN by mean value
df['peak-rpm'].replace(np.nan, avg_5, inplace= True)


''' This does not need mean becase options are either 2 door or 4 door
Therefore we will predict using the what value occurs the most
also known as frequency'''

df['num-of-doors'].value_counts()
df['num-of-doors'].value_counts().idxmax()

'''replace the missing 'num-of-doors' values by the most frequent'''
df["num-of-doors"].replace(np.nan, "four", inplace = True)

'''We are predicting price of car with vales missing
 We need to be as accurate as possible with price so we
 drop whole row with NaN in "price" column'''
df.dropna(subset=["price"], axis=0, inplace = True)

# reset index, because we droped two rows
df.reset_index(drop = True, inplace = True)

#print(df.head())


'''make sure all the data is in correct format'''
#print(df.dtypes)

'''Convert data to proper format'''

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
#print("Done")

#print(df.dtypes)

'''We need to assume that this data can be universally understood
therefore we will convert mpg to L/100km'''

#The formula for unit conversion is L/100km = 235 / mpg

# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# check your transformed data
#print(df.head())

df['highway-L/100km']=235/df['highway-mpg']

df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)
df.rename(columns={'"city-mpg"':'city-L/100km'}, inplace=True)
#print(df.head())

'''Data Normalization'''

# replace (origianl value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

# show the scaled columns
#print(df[["length","width","height"]].head())

'''Data Binning'''

#Binning Horsepower because we conly care about high, low, and medium horsepowers
#Convert data to proper format
df["horsepower"]=df["horsepower"].astype(float, copy=True)

#We would like four bins of equal size bandwidth,the forth is because the function "cut" include the rightmost edge:
binwidth = (max(df["horsepower"])-min(df["horsepower"]))/4
bins = np.arange(min(df["horsepower"]), max(df["horsepower"]), binwidth)
#print(bins)

group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names,include_lowest=True )
#print(df[['horsepower','horsepower-binned']].head(20))

'''Bin Visualization'''
a = (0,1,2)

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
#plt.pyplot.show()

'''Dummy Variables'''

#Fuel type only has gas or diesel so we can convert to indicator variables
#print(df.columns)


#Get indicator variables and assign it to data frame "dummy_variable_1"
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
#print(dummy_variable_1.head())

dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
#print(dummy_variable_1.head())

# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

#print(df.head())

# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])
dummy_variable_2.rename(columns={'turbo':'aspiration-turbo', 'std': 'aspiration-std'}, inplace=True)
#print(dummy_variable_2.head())

#merge the new dataframe to the original dataframe
df = pd.concat([df, dummy_variable_2], axis=1)
df.drop('aspiration', axis = 1, inplace=True)
df.to_csv("clean_df.csv")

