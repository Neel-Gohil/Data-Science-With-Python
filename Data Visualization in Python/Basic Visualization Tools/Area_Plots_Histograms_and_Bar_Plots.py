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
