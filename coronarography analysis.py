#!/usr/bin/env python
# coding: utf-8

# In[12]:


import math
import csv
import scipy as sp
from scipy import stats
import pandas as pd
import statistics
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import itertools


# In[13]:


"""linear regression: 
0. objective: understand the correlation between the passability and perfusion for each segment; 
1. take two number arrays for each instance of linreg: one from the 1st db, one from the 2d; issues: the number of arrays might not match; 
2. derive coefficient: it must be a positive correlation, perhaps set a threschold to check if the coefficient meets the expectations, i.e. high cag scores mean more pronounced hypoperfusion; 
3. lin reg between cag scores per pool vs ssr and sss"""


# In[43]:


df = pd.read_csv("coronarography")
df1 = pd.read_csv("2card - Sheet1.csv")

df1 = df1.rename(columns=lambda x: x[3:])
list1 = df1.columns.values.tolist()
newList = []
for i in list1:
    newList.append(i.split('.'))
newList = [x[::-1] for x in newList]
list2 = []
for j in newList:
    list2.append(''.join(j))
df1 = df1.rename(columns=dict(zip(df1.columns.values.tolist(), list2)))


# In[15]:


df.rename(columns={'ПМЖА устье':'1,2 ПМЖА устье, баз', 
                   'ПМЖА прокс':'1,2 ПМЖА прокс, баз',
                   'ПКА устье' : '3,4 ПКА устье баз',
                   'ПКА прокс' : '3,4 ПКА прокс баз',
                   #'ЗМЖА устье' : '3,4 ЗМЖА устье баз',
                   #'ЗМЖА прокс' : '3,4 ЗМЖА прокс',
                   'ОА устье' : '5,6 ОА устье',
                   'ОА прокс' : '5,6 ОА прокс',
                   'ПМЖА сред' : '7,8 ПМЖА сред',
                   'СВ1 устье' : '7,8 СВ1 устье сред',
                   'CВ 1 дист' : '7,8 CВ 1 дист сред',
                   'СВ2 устье' : '13,14, 19 СВ2 устье сред',
                   'ДА1 устье' : '7,8 ДА1 устье сред',
                   'ДА1 прокс' : '7,8 ДА1 прокс сред',
                   'ДА1 дист' : '7,8 ДА1 дист сред',
                   'ДА2 устье' : '13,14, 19 ДА2 устье сред',
                   'ДА2 прокс' : '13, 14, 19 ДА2 прокс сред',
                   'ДА2 средний' : '13, 14, 19 ДА2 средний сред',
                   'ДА2 дист' : '13,14,19 ДА2 лист сред',
                   'ПКА сред' : '3,4 ПКА сред',
                   'ЗМЖА сред' : '15,16 ЗМЖА сред',
                   'ПМЖАдист' : '13, 14, 19 ПМЖАдист',
                   'ЗМЖА дист' : '15,16 ЗМЖА дист',
                   'ЗБВ прокс' : '15,16 ЗБВ прокс',
                   'ЗБВ сред' : '15, 16 ЗБВ сред',
                   'ЗБЖ дист' : '15, 16 ЗБЖ дист',
                   'ПКА дист' : '9,10 ПКА дист',
                   'ЗМЖА прокс' : '9,10 ЗМЖА прокс',
                   'ОА средн' : '11,12 ОА средн',
                   'ВТК1 дист' : '11,12 ВТК1 дист',
                   'ВТК1 мед' : '11,12 ВТК1 мед',
                   'ВТК1 прокс' : '11,12 ВТК1 прокс',
                   'ВТК1 устье' : '11,12 ВТК1 устье',
                   'ВТК2 дист' : '11,12 ВТК2 дист',
                   'ВТК2 мед' : '11,12 ВТК2 мед',
                   'ВТК2 прокс' : '11,12 ВТК2 прокс',
                   'ВТК2 устье' : '11,12 ВТК2 устье'
                  },
          inplace=True)
df = df.reindex(sorted(df.columns), axis=1)
df = df.dropna(axis=1, how='all')
df = df.fillna("NaN")

for col in df.columns:
    unique = df[col].unique().tolist()
    if "NaN" in unique:
        unique.remove("NaN")
    if len(unique) == 1:
        df.drop(col,inplace=True,axis=1)
for column in df.columns:
    print(column)
df = df.drop(df.columns[19:], axis=1)


# In[16]:


coronaDataset = {}
for col in df:

    c = df[col]
    c = c.fillna('0')
    c1 = [str(elem) for elem in c]
    result = []
    for elem in c1:
        dashIndex = elem.find('-')
        if (dashIndex > -1): 
            elem = elem.split('-')
            elem = [float(i) for i in elem]
            elem = (elem[0] + elem[1]) / 2
        else:
            elem = float(elem)
        result.append(elem)

    coronaDataset[col] = result

regressionPairs = [
    ['1,2 ПМЖА прокс, баз', 'R1'],
    ['1,2 ПМЖА прокс, баз', 'R2'], 
    ['11,12 ВТК1 мед', 'R11'],
    ['11,12 ВТК1 мед', 'R12'],
    ['11,12 ВТК1 прокс', 'R11'],
    ['11,12 ВТК1 прокс', 'R12'],
    ['11,12 ВТК1 устье', 'R11'],
    ['11,12 ВТК1 устье', 'R12'],
    ['11,12 ВТК2 прокс', 'R11'],
    ['11,12 ВТК2 прокс', 'R12'],
    ['11,12 ОА средн', 'R11'],
    ['11,12 ОА средн', 'R12'],
    ['13, 14, 19 ПМЖАдист', 'R13'],
    ['13, 14, 19 ПМЖАдист', 'R14'],
    ['13, 14, 19 ПМЖАдист', 'R19'],
    ['15,16 ЗБВ прокс', 'R15'],
    ['15,16 ЗБВ прокс', 'R16'],
    ['3,4 ПКА прокс баз', 'R3'],
    ['3,4 ПКА прокс баз', 'R4'],
    ['3,4 ПКА сред', 'R3'],
    ['3,4 ПКА сред', 'R4'],
    ['5,6 ОА прокс', 'R5'],
    ['5,6 ОА прокс', 'R6'],
    ['7,8 ДА1 прокс сред', 'R7'],
    ['7,8 ДА1 прокс сред', 'R8'],
    ['7,8 ДА1 устье сред', 'R7'],
    ['7,8 ДА1 устье сред', 'R8'],
    ['7,8 ПМЖА сред', 'R7'],
    ['7,8 ПМЖА сред', 'R8'],
    ['7,8 СВ1 устье сред', 'R7'],
    ['7,8 СВ1 устье сред', 'R8'],
    ['9,10 ПКА дист', 'R9'],
    ['9,10 ПКА дист', 'R10'],
]

graphIndex = 0

for pair in regressionPairs:
    x_original = coronaDataset[pair[0]]
    y_original = df1[pair[1]].values.tolist()
    if len(x_original) != len(y_original):
        raise 'Everything is bad'
    x = []
    y = []
    for i in range(len(x_original)):
        if (not math.isnan(x_original[i])) and (not math.isnan(y_original[i])):
            x.append(x_original[i])
            y.append(y_original[i])
    
    
    x_regr = np.array(x).reshape((-1, 1))
    y_regr = np.array(y)
    model = LinearRegression().fit(x_regr, y_regr)
    
    r_sq = model.score(x_regr, y_regr)
    regressor = LinearRegression() 
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

    graphIndex = graphIndex + 1
    

    if slope > 0 and r_sq >= 0.7: 
        print (pair, "Number of observations", len(x_regr), 'Coefficient of determination:', r_sq, 'Positive correlation', slope, "P = ", p_value)
    if slope < 0 and r_sq >= 0.7: 
        print(pair, "Number of observations", len(x_regr), 'Coefficient of determination:', r_sq, 'Negative correlation', slope, "P = ", p_value)


# In[17]:


coronaDataset = {}
for col in df:
    c = df[col]
    c = c.fillna('0')
    c1 = [str(elem) for elem in c]
    result = []
    for elem in c1:
        dashIndex = elem.find('-')
        if (dashIndex > -1): 
            elem = elem.split('-')
            elem = [float(i) for i in elem]
            elem = (elem[0] + elem[1]) / 2
        else:
            elem = float(elem)
        result.append(elem)
    coronaDataset[col] = result
regressionPairs = [
    ['1,2 ПМЖА прокс, баз', 'S1'],
    ['1,2 ПМЖА прокс, баз', 'S2'], 
    ['11,12 ВТК1 мед', 'S11'],
    ['11,12 ВТК1 мед', 'S12'],
    ['11,12 ВТК1 прокс', 'S11'],
    ['11,12 ВТК1 прокс', 'S12'],
    ['11,12 ВТК1 устье', 'S11'],
    ['11,12 ВТК1 устье', 'S12'],
    ['11,12 ВТК2 прокс', 'S11'],
    ['11,12 ВТК2 прокс', 'S12'],
    ['11,12 ОА средн', 'S11'],
    ['11,12 ОА средн', 'S12'],
    ['13, 14, 19 ПМЖАдист', 'S13'],
    ['13, 14, 19 ПМЖАдист', 'S14'],
    ['13, 14, 19 ПМЖАдист', 'S19'],
    ['15,16 ЗБВ прокс', 'S15'],
    ['15,16 ЗБВ прокс', 'S16'],
    ['3,4 ПКА прокс баз', 'S3'],
    ['3,4 ПКА прокс баз', 'S4'],
    ['3,4 ПКА сред', 'S3'],
    ['3,4 ПКА сред', 'S4'],
    ['5,6 ОА прокс', 'S5'],
    ['5,6 ОА прокс', 'S6'],
    ['7,8 ДА1 прокс сред', 'S7'],
    ['7,8 ДА1 прокс сред', 'S8'],
    ['7,8 ДА1 устье сред', 'S7'],
    ['7,8 ДА1 устье сред', 'S8'],
    ['7,8 ПМЖА сред', 'S7'],
    ['7,8 ПМЖА сред', 'S8'],
    ['7,8 СВ1 устье сред', 'S7'],
    ['7,8 СВ1 устье сред', 'S8'],
    ['9,10 ПКА дист', 'S9'],
    ['9,10 ПКА дист', 'S10'],
]

graphIndex = 0

for pair in regressionPairs:
    x_original = coronaDataset[pair[0]]
    y_original = df1[pair[1]].values.tolist()
    if len(x_original) != len(y_original):
        raise 'Everything is bad'
    x = []
    y = []
    for i in range(len(x_original)):
        if (not math.isnan(x_original[i])) and (not math.isnan(y_original[i])):
            x.append(x_original[i])
            y.append(y_original[i])
    
    
    x_regr = np.array(x).reshape((-1, 1))
    #print(x_regr)
    y_regr = np.array(y)
    if not x_regr.any() or not y_regr.any():
        pass
    elif len(x_regr) < 2 or len(y_regr) < 2: 
        pass
    else:
        model = LinearRegression().fit(x_regr, y_regr)
        r_sq = model.score(x_regr, y_regr)
        regressor = LinearRegression() 
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

        if slope > 0 and r_sq >= 0.7: 
            print (pair, "Количество наблюдений", len(x_regr), 'Коэффициент детерминации:', r_sq, 'Позитивная корреляция', slope, "P = ", p_value) #est2.summary())
        if slope < 0 and r_sq >= 0.7: 
            print(pair, "Количество наблюдений", len(x_regr), 'Коэффициент детерминации:', r_sq, 'Негативная корреляция', slope, "P = ", p_value) #est2.summary())
    
    graphIndex = graphIndex + 1


# In[18]:

def columns_to_list(dataframe, first_letter):
    """this function creates a list of lists 
    based on a specific condition for the name of the column in df"""
    
    list_of_list = []
    list_of_col_names = [first_letter+str(x) for x in range(1,20)]
    for i in list_of_col_names:
        for col in dataframe:
            if col == i:
                list_of_list.extend(dataframe[col].tolist())
    return list_of_lists
        

R_list_of_all = columns_to_list(df1, 'R')
S_list_of_all = columns_to_list(df1, 'S')


R_list_of_lists = []
R_list_of_lists.append(R13_list)
R_list_of_lists.append(R14_list)
R_list_of_lists.append(R19_list)

S_list_of_lists = []
S_list_of_lists.append(S13_list)
S_list_of_lists.append(S14_list)
S_list_of_lists.append(S19_list)


def concatenate(list1, list2):
    merged_list = []
    for i in range(len(list1)):
        if (not math.isnan(list1[i])) and (not math.isnan(list2[i])):
            merged_list.append((list1[i] + list2[i])/2)
        elif (not math.isnan(list1[i])) and (math.isnan(list2[i])):
            merged_list.append(list1[i])
        elif math.isnan(list1[i]) and (not math.isnan(list2[i])):
            merged_list.append(list2[i])
        elif math.isnan(list1[i]) and math.isnan(list2[i]):
            merged_list.append(np.nan)
    return(merged_list)

def concatenate_multiple(list_multiple):
    merged_list_multiple = []
    zipped = list(zip(*list_multiple))
    for l in zipped:
        list_without_nans = list(filter(lambda x: not math.isnan(x), list(l)))
        if len(list_without_nans) == 0:
            merged_list_multiple.append(np.nan)
        elif len(list_without_nans) == 1:
            merged_list_multiple.append(list_without_nans[0])
        else:
            merged_list_multiple.append(statistics.mean(list_without_nans))
    return(merged_list_multiple)


merged_list_1_2 = concatenate(R1_list, R2_list)
merged_list_3_4 = concatenate(R3_list, R4_list)
merged_list_5_6 = concatenate(R5_list, R6_list)
merged_list_7_8 = concatenate(R7_list, R8_list)
merged_list_9_10 = concatenate(R9_list, R10_list)
merged_list_11_12 = concatenate(R11_list, R12_list)
merged_list_15_16 = concatenate(R15_list, R16_list)
merged_list_13_14_19 = concatenate_multiple(R_list_of_lists)

S_merged_list_1_2 = concatenate(S1_list, S2_list)
S_merged_list_3_4 = concatenate(S3_list, S4_list)
S_merged_list_5_6 = concatenate(S5_list, S6_list)
S_merged_list_7_8 = concatenate(S7_list, S8_list)
S_merged_list_9_10 = concatenate(S9_list, S10_list)
S_merged_list_11_12 = concatenate(S11_list, S12_list)
S_merged_list_15_16 = concatenate(S15_list, S16_list)
S_merged_list_13_14_19 = concatenate_multiple(S_list_of_lists)


# In[37]:


def difference(list1, list2):
    merged_list_dif = []
    for i in range(len(list1)):
        if not math.isnan(list1[i]) and not math.isnan(list2[i]):
            merged_list_dif.append(list1[i] - list2[i])
        elif math.isnan(list1[i]) or math.isnan(list2[i]):
            merged_list_dif.append(np.nan)
        elif math.isnan(list1[i]) and math.isnan(list2[i]):
            merged_list_dif.append(np.nan)
    return(merged_list_dif)

list_of_all_dif = []
for i,j in zip(R_list_of_all, S_list_of_all):
    list_of_all_dif.append(difference(i,j))


# In[41]:


regressionPairs = [
    ['1,2 ПМЖА прокс, баз', 1],
    ['1,2 ПМЖА прокс, баз', 2],
    ['11,12 ВТК1 мед', 11],
    ['11,12 ВТК1 мед', 12],
    ['11,12 ВТК1 прокс', 11],
    ['11,12 ВТК1 прокс', 12],
    ['11,12 ВТК1 устье', 11],
    ['11,12 ВТК1 устье', 12],
    ['11,12 ВТК2 прокс', 11],
    ['11,12 ВТК2 прокс', 12],
    ['11,12 ОА средн', 11],
    ['11,12 ОА средн', 12],
    ['13, 14, 19 ПМЖАдист', 13],
    ['13, 14, 19 ПМЖАдист', 14],
    ['13, 14, 19 ПМЖАдист', 17],
    ['15,16 ЗБВ прокс', 15],
    ['15,16 ЗБВ прокс', 16],
    ['3,4 ПКА прокс баз', 3],
    ['3,4 ПКА прокс баз', 4],
    ['3,4 ПКА сред', 3],
    ['3,4 ПКА сред', 4],
    ['5,6 ОА прокс', 5],
    ['5,6 ОА прокс', 6],
    ['7,8 ДА1 прокс сред', 7],
    ['7,8 ДА1 прокс сред', 8],
    ['7,8 ДА1 устье сред', 7],
    ['7,8 ДА1 устье сред', 8],
    ['7,8 ПМЖА сред', 7],
    ['7,8 ПМЖА сред', 8],
    ['7,8 СВ1 устье сред', 7],
    ['7,8 СВ1 устье сред', 8],
    ['9,10 ПКА дист', 9],
    ['9,10 ПКА дист', 10],
]
for pair in regressionPairs:
    x_original = coronaDataset[pair[0]]
    y_original = list_of_all_dif[pair[1] - 1]
    if len(x_original) != len(y_original):
        pass
    x = []
    y = []
    for i in range(len(x_original)):
        if (not math.isnan(x_original[i])) and (not math.isnan(y_original[i])):
            x.append(x_original[i])
            y.append(y_original[i])
    
    
    x_regr = np.array(x).reshape((-1, 1))
    y_regr = np.array(y)
    if not x_regr.any() or not y_regr.any():
        pass
    elif len(x_regr) < 2 or len(y_regr) < 2: 
        pass
    else:
        model = LinearRegression().fit(x_regr, y_regr)
        r_sq = model.score(x_regr, y_regr)
        regressor = LinearRegression() 
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

        if slope > 0 and r_sq >= 0.7: 
            print (pair, "Количество наблюдений", len(x_regr), 'Коэффициент детерминации:', r_sq, 'Позитивная корреляция', slope, "P = ", p_value) #est2.summary())
        if slope < 0 and r_sq >= 0.7: 
            print(pair,  "Количество наблюдений", len(x_regr), 'Коэффициент детерминации:', r_sq, 'Негативная корреляция', slope, "P = ", p_value) #est2.summary()) 


# In[ ]:




