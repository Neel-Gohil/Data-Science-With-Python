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

