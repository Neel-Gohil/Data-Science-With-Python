#Neel Gohil
#Basic Visualization Tools - Area Plots, Histograms, Bar Plots

from __future__ import print_function # this line adds compatibility to python 2
import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library

df_can = pd.read_excel('https://ibm.box.com/shared/static/lw190pt9zpy5bd1ptyg2aw15awomz9pu.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skipfooter=2
                      )

print('Data downloaded and read into a dataframe!')


print(df_can.head())


# print the dimensions of the dataframe
print(df_can.shape)

#clean up Data
df_can.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)

# first five elements and see how the dataframe was changed
print(df_can.head())


#Clean up column names so they are easier to understand
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)

# let's view the first five elements and see how the dataframe was changed
print(df_can.head())

# examine the types of the column labels
all(isinstance(column, str) for column in df_can.columns)


#Previous check came out false
#Change all column values to strings
df_can.columns = list(map(str, df_can.columns))

# let's check the column labels types now
all(isinstance(column, str) for column in df_can.columns)

#set country name as index for quicker searching
df_can.set_index('Country', inplace=True)

# view the first five elements and see how the dataframe was changed
df_can.head()

df_can['Total'] = df_can.sum(axis=1)
df_can.head()

print ('data dimensions:', df_can.shape)

# create a list of years from 1980 - 2014
# plotting the data will be easier
years = list(map(str, range(1980, 2014)))

years


'''Visualizing Data Using Matplotlib'''


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.style.use('ggplot') # optional: for ggplot-like style


#AREA PLOTS

df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)

# get the top 5 entries
df_top5 = df_can.head(5)

# transpose the dataframe
df_top5 = df_top5[years].transpose()

df_top5.head()

#Plot the data
df_top5.plot(kind='area',
             stacked=False,
             figsize=(20, 9), # pass a tuple (x, y) size
            )

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()

#HISTOGRAMS

#Question: What is the frequency distribution of the number (population) of new immigrants from the various countries to Canada in 2013?

#Look at 2013 data
df_can['2013'].head()

# np.histogram returns 2 values
count, bin_edges = np.histogram(df_can['2013'])

print(count) # frequency count
print(bin_edges) # bin ranges, default = 10 bins


df_can['2013'].plot(kind='hist', figsize=(8, 5))

plt.title('Histogram of Immigration from 195 Countries in 2013') # add a title to the histogram
plt.ylabel('Number of Countries') # add y-label
plt.xlabel('Number of Immigrants') # add x-label

plt.show()

#Question: What is the immigration distribution for Denmark, Norway, and Sweden for years 1980 - 2013?


#View Dataset
df_can.loc[['Denmark', 'Norway', 'Sweden'], years]

# generate histogram
df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
df_t.head()

#Better transparent histogram for better visualization
count, bin_edges = np.histogram(df_t, 15)

# Un-stacked Histogram
df_t.plot(kind ='hist',
          figsize=(10, 6),
          bins=15,
          alpha=0.6,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen']
         )

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()

'''Bar Charts'''

#Question: Let us compare the number of Icelandic immigrants (country = 'Iceland') to Canada from year 1980 to 2013.

# step 1: get the data
df_iceland = df_can.loc['Iceland', years]
df_iceland.head()


# step 2: plot data

df_iceland.plot(kind='bar', figsize=(10, 6))

plt.xlabel('Year') # add to x-label to the plot
plt.ylabel('Number of immigrants') # add y-label to the plot
plt.title('Icelandic immigrants to Canada from 1980 to 2013') # add title to the plot

plt.show()


#Question: Using the df_can dataset, create a horizontal bar plot showing the total number of immigrants to Canada from the top 15 countries, for the period 1980 - 2013.
# Label each country with the total immigrant count.

# sort dataframe on 'Total' column (descending)
df_can.sort_values(by='Total', ascending=True, inplace=True)

# get top 15 countries
df_top15 = df_can['Total'].tail(15)
df_top15

# generate plot
df_top15.plot(kind='barh', figsize=(12, 12), color='steelblue')
plt.xlabel('Number of Immigrants')
plt.title('Top 15 Conuntries Contributing to the Immigration to Canada between 1980 - 2013')

# annotate value labels to each country
for index, value in enumerate(df_top15):
    label = format(int(value), ',')  # format int with commas

    # place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
    plt.annotate(label, xy=(value - 47000, index - 0.10), color='white')

plt.show()