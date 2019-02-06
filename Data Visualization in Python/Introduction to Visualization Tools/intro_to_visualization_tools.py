#Neel Gohil
#Intro to Visualization Tools
#Going over basis of visualization

from __future__ import print_function # adds compatibility to python 2
import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import matplotlib as mpl
import matplotlib.pyplot as plt

'''Data Cleaning'''
#Read in Excel file found on ibm cloud storage
df_can = pd.read_excel('https://ibm.box.com/shared/static/lw190pt9zpy5bd1ptyg2aw15awomz9pu.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skipfooter=2)

#print ('Data read into a pandas dataframe!')

df_can.head()
df_can.tail()

#Get characteristics of dataset
df_can.info()

#Make columns and rows as lists for better manipulation if needed
df_can.columns.tolist()
df_can.index.tolist()

#print (type(df_can.columns.tolist()))
#print (type(df_can.index.tolist()))

#find dimensions of dataset
df_can.shape

#Remove unnecessary columns
# in pandas axis=0 represents rows (default) and axis=1 represents columns.
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)
df_can.head(2)

#rename the columns so that they make sense. Use rename() method by passing in a dictionary of old and new names as follows:
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
df_can.columns


#create a total column that sums up the total immigrants by country over the entire period 1980 - 2013, as follows:
df_can['Total'] = df_can.sum(axis=1)

#check to see how many null objects we have in the dataset as follows:
#print(df_can.isnull().sum())


#quick summary of each column in our dataframe using the describe() method.
print(df_can.describe())


'''Indexing and Slicing'''

#Selecting Column

df_can.Country  # returns a series

df_can[['Country', 1980, 1981, 1982, 1983, 1984, 1985]] #returns a dataframe

#Look at Japanese Immigration in multiple ways

# 1. the full row data (all columns)
df_can.loc['Japan']

# alternate methods
df_can.iloc[87]
df_can.ix[87]
df_can.ix['Japan']

# 2. for year 2013
df_can.loc['Japan', 2013]

# alternate methods
df_can.iloc[0, 36] # year 2013 is the last column, with a positional index of 36
df_can.ix['Japan', 36]

# does not work
# df_can.ix[87, 2013]

# 3. for years 1980 to 1985
df_can.ix[87, [1980, 1981, 1982, 1983, 1984, 1984]]

# alternate methods
df_can.loc['Japan', [1980, 1981, 1982, 1983, 1984, 1984]]
df_can.iloc[87, [3, 4, 5, 6, 7, 8]]

#To avoid this ambuigity, let's convert the column names into strings: '1980' to '2013'

df_can.columns = list(map(str, df_can.columns))


#declare a variable that will allow us to easily call upon the full range of years:
# useful for plotting later on
years = list(map(str, range(1980, 2014)))

'''Filtering based on criteria'''

#filter the dataframe to show the data on Asian countries (AreaName = Asia)

# 1. create the condition boolean series
condition = df_can['Continent']=='Asia'
print (condition)

# 2. pass this condition into the dataFrame
df_can[condition]

# we can pass mutliple criteria in the same line.
# let's filter for AreaNAme = Asia and RegName = Southern Asia

df_can[(df_can['Continent']=='Asia') & (df_can['Region']=='Southern Asia')]


#review the changes we have made to our dataframe
print ('data dimensions:', df_can.shape)
print(df_can.columns)
df_can.head(2)

'''Visualizing using Matplotlib'''

#Line Plots

#Case Study:
# In 2010, Haiti suffered a catastrophic magnitude 7.0 earthquake.
# The quake caused widespread devastation and loss of life and aout three million people were affected by this natural disaster.
# As part of Canada's humanitarian effort, the Government of Canada stepped up its effort in accepting refugees from Haiti.
# Find the immigration rate of Haitians to Canada over time

haiti = df_can.loc['Haiti', years] # Passing in years 1980 - 2013 to exclude the 'total' column
haiti.head()

#plots the values in haiti datafrome
haiti.plot()

#label the x and y axis using plt.title(), plt.ylabel(), and plt.xlabel() as follows:
haiti.plot(kind='line')

plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')

plt.show() # Need this line to show the updates made to the figure

#Aft, r earthquake immigrants from Haiti spiked up from 2010 as Canada stepped up its efforts to accept refugees from Haiti.
#annotate this spike in the plot by using the plt.text() method.

# Since the x-axis (years) is type 'string', we need to specify the years in terms of its index position. Eg 20th index is year 2000.
# The y axis (number of Immigrants) is type 'integer', so we can just specify the value y = 6000.
# plt.text(20, 6000, '2010 Earthquake') # years stored as type str
# If the years were stored as type 'integer' or 'float', we would have specified x = 2000 instead.
# plt.text(2000, 6000, '2010 Earthquake') # years stored as type int
# We will cover advanced annotation methods in later modules.

#Question: Let us compare the number of immigrants from India and China from 1980 to 2013.


#Get Data
df_CI = df_can.loc[['India', 'China'], years]
df_CI.head()

#Plot kind of plot: Line
df_CI.plot(kind='line')

df_CI = df_CI.transpose()
df_CI.head()

df_CI.plot(kind='line')

plt.title('Immigrants from China and India')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()


#Question: Compare the trend of top 5 countries that contributed the most to immigration to Canada.

# inplace = True paramemter saves the changes to the original df_can dataframe
df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)

# get the top 5 entries
df_top5 = df_can.head(5)

# transpose the dataframe
df_top5 = df_top5[years].transpose()

#print(df_top5)

df_top5.plot(kind='line', figsize=(14, 8)) # pass a tuple (x, y) size

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()