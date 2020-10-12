#!/usr/bin/env python
# coding: utf-8

# In[10]:


import math
import csv
import scipy as sp
import pandas as pd
import statistics
import numpy as np
from scipy import stats
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt


# In[11]:


df = pd.read_csv("descr - Sheet1.csv")
df1 = pd.read_csv("card data.csv")
dummy = pd.get_dummies(df['CД'])
dummy = pd.get_dummies(df['ИМ в анамнезе'])
print(df)
df.fillna(0, inplace = True)
df1.fillna(0, inplace = True)


# In[12]:


def sum_individual(dataframe, firstletter):
    sum_individual = []
    for row in dataframe.iterrows():
        score = 0
        count = 0
        for column in row[1].keys():
            if column.startswith(firstletter):
                value = row[1][column]
                if value == 0:
                    continue
                count += 1
                if value > 80:
                    score += 0
                elif value >= 75 and value <= 80:
                    score += 1
                elif value >= 50 and value <= 74:
                    score += 2
                elif value >= 25 and value <= 49:
                    score += 3
                else:
                    score += 4
        if count > 0:
            sum_individual.append(score)
        else:
            sum_individual.append(None)

    return sum_individual


# In[ ]:


SSD = []
for i, j in zip(sum_individual('R'), sum_individual('S')):
    if (i is None) or (j is None):
        SSD.append(None)
    else:
        SSD.append(i - j)


# In[1]:


def boxplot(dataframe, column, criterion):
    sns.set(style="ticks", color_codes=True)
    
    df2 = dataframe.assign(Column = dataframe[column])
    df2 = df2.assign(Condition = criterion)
    df2 = df2.drop(df2[df2.Condition == 0].index)

    sns.boxplot(x=column, y=Condition, data=df2)
    df3 = pd.DataFrame(
        {
            "Column": df2.Column,
            "Condition": df2.Condition
        }
    )
    df3 = df3.dropna()
    SD = df3["Column"].tolist()
    RS = df3['Condition'].tolist()
    M = np.cov(SD, RS)
    return M[0][1]


# In[7]:


#0. goal: to find a heart segment that shows the highest correlation with hypertension (H)
#1. we iterate over columns in df1 and each time concatenate a column and H together for finding correlation
#2. all the coefficients are stored in list_of_coeff
#3. then the list is sorted in the descending order, keeping the ordinal number of each segment


# In[37]:


def ranking_covariation(dataframe, first_letter, determinant):
    list_of_coeff = []
    list_of_indices = []
    for column in dataframe: 
        if column.startswith(first_letter):
            df2 = df1.assign(determinant = dataframe[determinant])
            df2 = df2.assign(first_letter = dataframe[column])
            df3 = pd.DataFrame(
                {
                    determinant: df2.determinant,
                    first_letter: dataframe[column]
                }
            )
            df3 = df3.dropna()
            determinant_list = df3[determinant].tolist()
            condition_list = df3[firstletter].tolist()
            matrix = np.cov(determinant_list, condition_list)
            list_of_coeff.append(matrix[0][1])
            list_of_indices.append(dataframe.columns.get_loc(column)/2)  
    return sorted(tuple(zip(list_of_coeff, list_of_indices)), key=lambda x: x[0]) 

