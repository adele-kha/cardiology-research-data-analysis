#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from pandas import *
from scipy import stats
import scipy.stats as stats
import math
from itertools import islice
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import statsmodels.api as sm

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import itertools
import collections
from collections import OrderedDict


# In[18]:


#reading and cleaning
f = pd.read_csv('scinti.csv')
f.columns = f.iloc[0]
f = f.loc[1:]
f.fillna(0, inplace = True)
f.rename(columns=lambda x: x[3:], inplace=True)
list1 = f.columns.values.tolist()
newList = []
for i in list1:
    newList.append(i.split('.'))
newList = [x[::-1] for x in newList]
list2 = []
for j in newList:
    list2.append(''.join(j))
f = f.rename(columns=dict(zip(f.columns.values.tolist(), list2)))


# In[19]:


def list_clean(column):
    return list(map(lambda x: float(x), f[column].values.tolist()))

r_list_13 = list_clean('R13')
r_list_14 = list_clean('R14')
r_list_15 = list_clean('R15')
r_list_16 = list_clean('R16')
r_list_17 = list_clean('R17')
r_list_18 = list_clean('R18')

r_unified_list_13 = list(map(lambda y: sum(y) / len(y), zip(r_list_13, r_list_14, r_list_15)))
r_unified_list_13 = [float("%.2f" % x) for x in r_unified_list_13]
r_unified_list_14 = list(map(lambda y: sum(y) / len(y), zip(r_list_14, r_list_15)))
r_unified_list_14 = [float("%.2f" % x) for x in r_unified_list_14]
r_unified_list_15 = list(map(lambda y: sum(y) / len(y), zip(r_list_15, r_list_16, r_list_17)))
r_unified_list_15 = [float("%.2f" % x) for x in r_unified_list_15]
r_unified_list_16 = list(map(lambda y: sum(y) / len(y), zip(r_list_18, r_list_17)))
r_unified_list_16 = [float("%.2f" % x) for x in r_unified_list_16]


f['R13_1'] = r_unified_list_13
f['R14_1'] = r_unified_list_14
f['R15_1'] = r_unified_list_15
f['R16_1'] = r_unified_list_16


# In[20]:


s_list_13 = list_clean('S13')
s_list_14 = list_clean('S14')
s_list_15 = list_clean('S15')
s_list_16 = list_clean('S16')
s_list_17 = list_clean('S17')
s_list_18 = list_clean('S18')

s_unified_list_13 = list(map(lambda y: sum(y) / len(y), zip(s_list_13, s_list_14, s_list_15)))
s_unified_list_13 = [float("%.2f" % x) for x in s_unified_list_13]
s_unified_list_14 = list(map(lambda y: sum(y) / len(y), zip(s_list_14, s_list_15)))
s_unified_list_14 = [float("%.2f" % x) for x in s_unified_list_14]
s_unified_list_15 = list(map(lambda y: sum(y) / len(y), zip(s_list_15, s_list_16, s_list_17)))
s_unified_list_15 = [float("%.2f" % x) for x in s_unified_list_15]
s_unified_list_16 = list(map(lambda y: sum(y) / len(y), zip(s_list_18, s_list_17)))
s_unified_list_16 = [float("%.2f" % x) for x in s_unified_list_16]


f['S13_1'] = s_unified_list_13
f['S14_1'] = s_unified_list_14
f['S15_1'] = s_unified_list_15
f['S16_1'] = s_unified_list_16


# In[21]:


Fr = f

Fr.drop(['', 'e', 'x', 'type', '', 'abetes', 'pertension', 'VR', 'VS','LVR','LVS','pression ST','pression ST max','mber of leads','poperfusion'], axis=1, inplace = True)
Fr.drop(['R13', 'R14', 'R15', 'R16', 'S13', 'S14', 'S15', 'S16', 'R17', 'S17', 'R18', 'S18' ], axis = 1, inplace = True)

cols = Fr.columns.tolist()

cols.insert(24, cols.pop(cols.index('R13_1')))
cols.insert(25, cols.pop(cols.index('S13_1')))
cols.insert(26, cols.pop(cols.index('R14_1')))
cols.insert(27, cols.pop(cols.index('S14_1')))
cols.insert(28, cols.pop(cols.index('R15_1')))
cols.insert(29, cols.pop(cols.index('S15_1')))
cols.insert(30, cols.pop(cols.index('R16_1')))
cols.insert(31, cols.pop(cols.index('S16_1')))

Fr.columns = cols
print(Fr)


# In[22]:


#saving as a csv file
Fr.to_csv('only_segments_scinti.csv') 
Fr.to_excel('only_segments_scinti.xlsx')


# In[23]:


list_of_lists = Fr.to_numpy().tolist()
list_of_lists = [list(map(float, sublist)) for sublist in list_of_lists]
list_of_deltas = []
for l in list_of_lists:
    l = list(zip(l[::2], l[1::2]))
    l = [tup[1] - tup [0] for tup in l]
    l = [float("%.2f" % x) for x in l]
    list_of_deltas.append(l)
    
print(len(list_of_deltas))
list_of_deltas.pop()
list_of_deltas.pop()
#list_of_deltas_f = list_of_deltas_f.pop(20)
print(len(list_of_deltas))
print(list_of_deltas)


# In[24]:


df = pd.read_csv('scinti_big.csv')
df.columns = df.iloc[0]
df = df.loc[1:]


df1 = pd.read_csv('scinti.csv') 
df1.columns = df1.iloc[0]
df1 = df1.loc[1:]

col_df = [col for col in df.columns]
col_df1 = [col for col in df1.columns]
print(col_df)
indices_all = df[' №'].values.tolist()
indices_some = df1[' №'].values.tolist()
age = df[' age'].values.tolist()
indices_some.pop()
indices_some.pop()


dict_all_age = dict(zip(indices_all, age))


ages_complete = []
ages_incomplete = []
for key, value in dict_all_age.items():
    if key in indices_some:
        ages_complete.append(float(value))
    else:
        ages_incomplete.append(float(value))
        
print(ages_complete, ages_incomplete)


# In[25]:


#for tests!
ttest_age = stats.ttest_ind(ages_complete,ages_incomplete)
print(ttest_age)


# In[2]:


def fischer_test(column):
    column_to_list = df[' sex'].values.tolist()
    if not all(characters.isalpha() for characters in column_to_list):
        column_to_list = [float(x) for x in column_to_list]
        column_to_list = [0 if math.isnan(x) else x for x in column_to_list]
    dict_all = dict(zip(indices_all, column_to_list))
    column_to_list_complete = []
    column_to_list_incomplete = []
    for key, value in dict_all.items():
        if key in indices_some:
            column_to_list_complete.append(value)
        else:
            column_to_list_incomplete.append(value)
    set_column_to_list_complete = list(set(column_to_list_complete))
    first_element_complete = sex_complete.count(set_column_to_list_complete[0])
    second_element_complete = sex_complete.count(set_column_to_list_complete[1])
    first_element_incomplete = sex_incomplete.count(set_column_to_list_complete[0])
    second_element_incomplete = sex_incomplete.count(set_column_to_list_complete[1])
    list_for_fisher = np.array([[first_element_complete,second_element_complete,first_element_incomplete,second_element_incomplete]])
    list_for_fisher = np.reshape(list_for_fisher,(2,2))
    oddsratio, pvalue = stats.fisher_exact(list_for_fisher)
    return "oddsratio:", oddsratio, ", pvalue:", pvalue


# In[32]:


# 1. there are 17 segments in the ventricle: [1..17]
# 2. they belong to different parts, for example, [0..5] are basal, [5..10] are medial etc. 
# 3. it is possible that within one ventricle, different parts show different malfunction
# 4. to understand which segment shows which malfunction, you need to read the diagnosis
# 5. diagnoses do not explain each segment, they explain parts (basal, medial etc), which allows to assign the same category to a group of segments
# 6. if a diagnose does not mention the stages at all, then all segments == 1 (normal)
# 7. if a diagnose mentions stage, then segments can be assigned values from two to four


# In[33]:


md = pd.read_csv("medical_texts.csv")
md.columns = md.iloc[0]
md = md.loc[1:]
texts = md['Заключение1'].values.tolist()
indices = md['№'].values.tolist()
indices = [float(i) for i in indices]

"""class TextProc():
    def __init__(self):
        self.text = text
    def text_tok(self, text):
        tokenizer = RegexpTokenizer(r'\w+')
        text = [tokenizer.tokenize(word) for word in text]
        text = list(filter(None, text))
        return text
    def text_stem(self, text): 
        stemmer = SnowballStemmer("russian") 
        text = [stemmer.stem(word) for word in text]
        return text

text = TextProc()
texts_tok = text.text_tok(texts)
texts_stem = [text.text_stem(t) for t in texts_tok]"""

texts_dict = dict(zip(indices, texts))


# In[34]:


list_of_stages = ["нормокинез", "гипокинез", "акинез", "дискинез"]

keys = []
list_of_list_of_segments = []
for key, value in texts_dict.items():
    if not any(word in value for word in list_of_stages):
        list_of_segments = [1] * 17
        keys.append(key)
        list_of_list_of_segments.append(list_of_segments)
        dict_final = dict(zip(keys, list_of_list_of_segments))

for key, value in texts_dict.items():
    if not key in keys:
        value_tok = nltk.tokenize.sent_tokenize(value)
        for sent in value_tok:
            if any(word in sent for word in list_of_stages):
                print(key, sent)
            

dict_final[3] = [1,1,1,2,2,1,1,1,1,2,2,2,1,2,2,2,1]
dict_final[7] = [1,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1]
dict_final[6] = [1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1]
dict_final[17] = [1,2,2,2,1,1,1,1,1,2,2,1,1,1,1,1,1]
dict_final[24] = [2,2,2,2,2,2,1,1,1,2,2,2,1,1,1,1,1]
dict_final[39] = [1,2,2,3,2,1,1,1,1,3,2,1,1,1,1,1,1]
dict_final[40] = [2,1,1,1,1,1,2,3,1,1,2,1,3,3,3,3,1]
dict_final[42] = [1,1,2,1,1,1,1,1,1,2,3,3,3,1,2,3,1]
dict_final[49] = [1,1,2,2,2,1,1,1,1,2,1,1,1,1,1,1,1]
dict_final[51] = [1,1,1,1,1,1,1,2,2,1,1,1,1,2,2,1,1]
dict_final[53] = [1,2,2,2.5,1,1,1,1,2,2.5,2,1,2,2,2,2,1]

dict_final = collections.OrderedDict(sorted(dict_final.items()))
print(dict_final)


# In[35]:


list_of_echos = list(dict_final.values())
list_of_indices = list(dict_final.keys())

data_list = []
for i,j,k in zip(list_of_deltas, list_of_echos, list_of_indices):
    data = {"Номер пациента":k,'SSD':i, 'Echo':j}
    df = pd.DataFrame(data)
    df.loc['Mean'] = df.mean()
    data_list.append(df)

df_all = pd.concat(data_list)
df_all.to_excel("pat.xlsx")


# In[214]:


df1 = pd.read_csv("scinti+echo+kag updated.csv")

def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

list_of_ssd = df1["SSD"].values.tolist()
list_of_ssd = [i.replace(',','.') if ',' in i else i for i in list_of_ssd]
list_of_ssd = [float(i) for i in list_of_ssd]
list_of_ssd = list(divide_chunks(list_of_ssd, 18))


list_of_echo = df1["Echo"].values.tolist()
list_of_echo = [i.replace(',','.') if ',' in i else i for i in list_of_echo]
list_of_echo = [float(i) for i in list_of_echo]
list_of_echo = list(divide_chunks(list_of_echo,18))


list_of_patients_dataframes = [] 
for i,j,k in zip(list_of_ssd, list_of_echo, list_of_indices):
    data = {"Номер пациента":k,'SSD':i, 'Echo':j}
    df = pd.DataFrame(data)
    list_of_patients_dataframes.append(df)
    
list_for_segments = []
for k in range(18):
    for i in range(21): 
        x = list_of_patients_dataframes[i]
        list_for_segments.append(x.iloc[k].values.tolist())

list_for_segments = list(divide_chunks(list_for_segments,21))

def prepare_for_reg_pred(biglist):
    for smalllist in biglist:
        for k in smalllist:
            yield k[1]
            
def prepare_for_reg_out(biglist):
    for smalllist in biglist:
        for k in smalllist:
            yield k[2]

list_of_pred = list(prepare_for_reg_pred(list_for_segments))
list_of_pred = list(divide_chunks(list_of_pred, 21))
list_of_pred = list_of_pred[:-1]


list_of_out = list(prepare_for_reg_out(list_for_segments))
list_of_out = list(divide_chunks(list_of_out, 21))
list_of_out = list_of_out[:-1]
print(list_of_out)


# In[226]:


for idx, (i, j) in enumerate(zip(list_of_pred, list_of_out), start=1):
    i_regr = np.array(i).reshape((-1, 1))
    j_regr = np.array(j).reshape((-1, 1))
    model = LinearRegression().fit(i_regr, j_regr)
    r_sq = model.score(i_regr, j_regr)
    regressor = LinearRegression() 
    slope, intercept, r_value, p_value, std_err = stats.linregress(i,j)
    if slope > 0: 
        print (idx,'Coefficient of determination:', r_sq, 'Positive correlation', slope, "P = ", p_value) 
    if slope < 0: 
        print(idx, 'Coefficient of determination:', r_sq, 'Negative correlation', slope, "P = ", p_value)


# In[ ]:




