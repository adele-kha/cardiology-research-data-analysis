#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import csv
import scipy as sp
import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model


# In[2]:

"""opening the file and applying one hot encoding"""

df = pd.read_csv("descr - Sheet1.csv")
dummy = pd.get_dummies(df['CД'])
df.fillna(0, inplace = True)

def sample_statistics(dataframe, column):
    """this function finds the percentage of patients 
    with a certain characteristic such as sex, diabetes, 
    myocardial infarction, and others"""
    
    unique_elem = set(dataframe[column].tolist())
    number_of_first_elem = len(dataframe[dataframe[column] == unique_elem[0]])
    number_of_second_elem = len(dataframe[dataframe[column] == unique_elem[1]])
    total_count = number_of_first_elem + number_of_second_elem 
    
    return unique_elem[0], number_of_first_elem/total_count*100, unique_elem[1], number_of_second_elem/total_count*100,

mean_age = "Mean age", df["возраст"].mean()
age_std = df["возраст"].std()


df = pd.read_csv("card data.csv")


allRMeans = []
allSMeans = []
for column in df:
    mean = df[column].mean()
    if column.startswith("R"):
        allRMeans.append(mean)
    if column.startswith("S"):
        allSMeans.append(mean)

RMeansMalfunctioning = list(filter(lambda x: x <= 80, allRMeans)) 
SMeansMalfunctioning = list(filter(lambda x: x <= 80, allSMeans))
print("Number of segments with hypoperfusion at rest", len(RMeansMalfunctioning))
print("Number of segments with hypoperfusion in stress", len(SMeansMalfunctioning))


df.fillna(0, inplace = True)

def info_per_patient(dataframe, first_letter):
    """this function calculates scintigraphy hypoperfusion scores
    per patient and provides general numbers regarding the number
    of cases per level of severity"""
    
    number_of_no_threat = 0
    number_of_mild_threat = 0
    number_of_medium_threat = 0
    number_of_grave_cases = 0
    number_of_nonzero_patients = 0
    for row in dataframe.iterrows():
        score = 0
        count = 0
        for column in row[1].keys():
            if column.startswith(first_letter):
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
            number_of_nonzero_patients += 1
            print(row[0], "Patient scores at rest = ", score)
        else:
            print(row[0], "Patient scores at rest","No information")
        if score == 0:
            continue
        if score < 4:
            number_of_no_threat += 1
            print("No risk")
        elif score >= 4 and score<=7:
            number_of_mild_threat += 1
            print("mild violation of myocardial blood flow")
        elif score >= 8 and score <=11:
            number_of_medium_threat += 1
            print("moderate severity of hypoperfusion")
        elif score >= 12:
            number_of_grave_cases += 1
    print ("severe myocardial perfusion disorders and high risk of coronary complications")
    print("percentage of severely impaired patients = ", number_of_grave_cases / number_of_nonzero_patients * 100)
    return number_of_no_threat, number_of_mild_threat, number_of_medium_threat, number_of_grave_cases




#patient data based on the severity of disorder
objects = ('No risk', 'Mild violation of myocardial blood flow', 'Moderate severity of hypoperfusion', 'Severe myocardial perfusion disorders')
patients_under_stress = info_per_patient(df, 'S')
patients_at_rest = info_per_patient(df, 'R')



def number_per_degree(lis):
    """this function produces a bar chart 
    with number of patients per level of severity"""
    
    y_pos = np.arange(len(objects))
    barWidth = 0.9

    plt.bar(y_pos, lis, align = 'center', alpha = 0.5)
    plt.xticks (y_pos, objects, rotation=90)
    plt.ylabel('Number of patients')
    plt.title('Number of patients on the basis of disorder severity')
    for index,data in enumerate(lis):
        plt.text(x=index , y =data + 0.5, s=f"{data}" , fontdict=dict(fontsize=10))
    plt.ylim(0, max(lis) + 2)
    plt.show()


data = [patients_under_stress, patients_at_rest]

X = np.arange(4)
plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)
plt.bar(X + 0.25, data[1], color = 'g', width = 0.25)
y_pos = np.arange(len(objects))

plt.xticks (y_pos, objects, rotation=90)

for index,data in enumerate(patients_under_stress):
    plt.text(x=index , y =data + 0.5, s=f"{data}" , fontdict=dict(fontsize=10))
for index, data in enumerate(patients_at_rest):
    plt.text(x=index , y =data + 0.5, s=f"{data}" , fontdict=dict(fontsize=10))
plt.title('At rest vs under stress in absolute values')
plt.show()


def percent_per_degree_pie(lis):
    """this function produces a pie chart with patient percentages
    per level of severity"""
    
    patients_in_percents = []
    for i in lis:
        percent = i / number_of_nonzero_patients * 100
        patients_in_percents.append(percent)
    #patients_in_percents = [i for i in patients_in_percents if i != 0.0]
    plt.pie(patients_in_percents,labels=objects,autopct='%1.1f%%')
    plt.title('Процент пациентов по степени нарушений в стрессе')
    plt.axis('equal')
    plt.show()
