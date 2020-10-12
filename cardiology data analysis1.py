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


df = pd.read_csv("descr - Sheet1.csv")
dummy = pd.get_dummies(df['CД'])
df.fillna(0, inplace = True)

numberOfMales = len(df[df['пол'] == 'м'])
numberOfFemales = len(df[df['пол'] == 'ж'])
totalCount = numberOfMales + numberOfFemales

numberOfDiabetes = len(df[df['CД'] == 1])
numberOfNoDiabetes = len(df[df['CД'] == 0])
totalCount1 = numberOfDiabetes + numberOfNoDiabetes

numberOfInfarction = len(df[df['ИМ в анамнезе'] == 1])
numberOfNoInfarction = len(df[df['ИМ в анамнезе'] == 0])
totalCount2 = numberOfInfarction + numberOfNoInfarction

numberOfHypertension = len(df[df['ГБ'] == 1])
numberOfNoHypertension = len(df[df['ГБ'] == 0])
totalCount3 = numberOfHypertension + numberOfNoHypertension

print("Percentage of men, % = ", numberOfMales / totalCount * 100)
print("Percentage of women, % = ", numberOfFemales / totalCount * 100)
print("Percentage of diabetes patients, % = ", numberOfDiabetes / totalCount1 * 100)
print("Percentage of patients without diabetes, % = ", numberOfNoDiabetes / totalCount1 * 100)
print("Percentage of patients with myocardial infarction, % = ", numberOfInfarction / totalCount2 * 100)
print("Percentage of patients without myocardial infarction, % = ", numberOfNoInfarction / totalCount2 * 100)
print("Percentage of patients with myocardial hypertension, % = ", numberOfHypertension / totalCount3 * 100)
print("Percentage of patients without myocardial hypertension, % = ", numberOfNoHypertension / totalCount2 * 100)


# In[3]:


print("Mean age", df["возраст"].mean(), "+-", df["возраст"].std(), "years old")


# In[4]:


df = pd.read_csv("card data.csv")


# In[5]:


df.fillna(0, inplace = True)
r_non_zero_patient_count = 0
total_sum = 0

for row in df.iterrows():
    sum_individual = 0
    r_non_zero = False
    for column in row[1].keys():
        if column.startswith("R"):
            value = row[1][column]
            if value == 0:
                continue

            r_non_zero = True

            if value < 80:
                sum_individual += 1
    if r_non_zero:
        r_non_zero_patient_count += 1 

    total_sum += sum_individual
s_non_zero_patient_count = 0  
s_total_sum = 0
for row in df.iterrows():
    s_sum_individual = 0
    s_non_zero = False
    for column in row[1].keys():
        if column.startswith("S"):
            value = row[1][column]
            if value == 0:
                continue

            s_non_zero = True

            if value < 80:
                s_sum_individual += 1
    if s_non_zero:
        s_non_zero_patient_count += 1 

    s_total_sum += s_sum_individual

print("Sum of all sums (S)", total_sum)
print("Number of non-zero patients (S)", r_non_zero_patient_count)
print("Sum all sums and divide by the number of patients (S)", total_sum / r_non_zero_patient_count)
print("Sum of all sums (S)", s_total_sum)
print("Number of non-zero patients (S)", s_non_zero_patient_count)
print("Sum all sums and divide by the number of patients (S)", s_total_sum / s_non_zero_patient_count)


# In[6]:


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


# In[23]:


df = pd.read_csv("card data.csv")
df.fillna(0, inplace = True)

def info_per_patient(dataframe, first_letter):
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
            print(row[0], "Patient scores = ", score)
        else:
            print(row[0], "Patient scores = ", "No information")
        if score == 0:
            continue
        if score < 4:
            print("No risk")
        elif score >= 4 and score<=7:
            print("mild violation of myocardial blood flow")
        elif score >= 8 and score <=11:
            print("moderate severity of hypoperfusion")
        elif score >= 12:
            print ("severe myocardial perfusion disorders and high risk of coronary complications")


# In[24]:


print(info_per_patient(df, "S"))


# In[29]:


df.fillna(0, inplace = True)

def info_per_patient(dataframe, first_letter):
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


# In[30]:


print(info_per_patient(df, "S"))


# In[34]:


#patient data based on the severity of disorder
objects = ('No risk', 'Mild violation of myocardial blood flow', 'Moderate severity of hypoperfusion', 'Severe myocardial perfusion disorders')
patients_under_stress = info_per_patient(df, 'S')
patients_at_rest = info_per_patient(df, 'R')


# In[35]:


def number_per_degree(lis):
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


# In[36]:


print(number_per_degree(patients_under_stress))


# In[37]:


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


# In[20]:


def percent_per_degree_pie(lis):
    patients_in_percents = []
    for i in lis:
        percent = i / number_of_nonzero_patients * 100
        patients_in_percents.append(percent)
    #patients_in_percents = [i for i in patients_in_percents if i != 0.0]
    plt.pie(patients_in_percents,labels=objects,autopct='%1.1f%%')
    plt.title('Процент пациентов по степени нарушений в стрессе')
    plt.axis('equal')
    plt.show()


# In[21]:


print(percent_per_degree_pie(patients_under_stress))

