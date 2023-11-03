
# coding: utf-8

# # Statistic for Data Science Final Exam

# In[16]:


import pandas as pd


# In[17]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url, header= 0)


# In[18]:


boston_df.head


# <bound method NDFrame.head of      Unnamed: 0     CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD  \
# 0             0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0   
# 1             1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0   
# 2             2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0   
# 3             3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0   
# 4             4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0   
# ..          ...      ...   ...    ...   ...    ...    ...   ...     ...  ...   
# 501         501  0.06263   0.0  11.93   0.0  0.573  6.593  69.1  2.4786  1.0   
# 502         502  0.04527   0.0  11.93   0.0  0.573  6.120  76.7  2.2875  1.0   
# 503         503  0.06076   0.0  11.93   0.0  0.573  6.976  91.0  2.1675  1.0   
# 504         504  0.10959   0.0  11.93   0.0  0.573  6.794  89.3  2.3889  1.0   
# 505         505  0.04741   0.0  11.93   0.0  0.573  6.030  80.8  2.5050  1.0   
# 
#        TAX  PTRATIO  LSTAT  MEDV  
# 0    296.0     15.3   4.98  24.0  
# 1    242.0     17.8   9.14  21.6  
# 2    242.0     17.8   4.03  34.7  
# 3    222.0     18.7   2.94  33.4  
# 4    222.0     18.7   5.33  36.2  
# ..     ...      ...    ...   ...  
# 501  273.0     21.0   9.67  22.4  
# 502  273.0     21.0   9.08  20.6  
# 503  273.0     21.0   5.64  23.9  
# 504  273.0     21.0   6.48  22.0  
# 505  273.0     21.0   7.88  11.9  
# 
# [506 rows x 14 columns]>

# # 1. Median value of owner-occupied homes

# In[19]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.boxplot(boston_df['MEDV'])
plt.xlabel('Median Value of Owner-Occupied Homes')
plt.ylabel('Value')
plt.title('Boxplot of Median Value of Owner-Occupied Homes')
plt.show()


# # 2. Charles river variable

# In[20]:


import matplotlib.pyplot as plt

# Count the occurrences of each value in the Charles river variable
charles_river_counts = boston_df['CHAS'].value_counts()

# Create a bar plot
plt.figure(figsize=(6, 4))
charles_river_counts.plot(kind='bar')
plt.xlabel('Charles River')
plt.ylabel('Count')
plt.title('Bar Plot of Charles River')
plt.xticks(rotation='horizontal')
plt.show()


# # 3. Boxplot of MEDV vs AGE

# In[21]:


import matplotlib.pyplot as plt

# Discretize the AGE variable into three groups
boston_df['AGE_GROUP'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, float('inf')], labels=['35 years and younger', 'between 35 and 70 years', '70 years and older'])

# Create the boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([boston_df[boston_df['AGE_GROUP'] == '35 years and younger']['MEDV'],
             boston_df[boston_df['AGE_GROUP'] == 'between 35 and 70 years']['MEDV'],
             boston_df[boston_df['AGE_GROUP'] == '70 years and older']['MEDV']],
            labels=['35 years and younger', 'between 35 and 70 years', '70 years and older'])
plt.xlabel('Age Group')
plt.ylabel('Median Value of Owner-Occupied Homes')
plt.title('Boxplot of MEDV vs AGE')
plt.show()


# # 4. Scatter Plot: NOX vs INDUS

# In[22]:


import matplotlib.pyplot as plt

# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(boston_df['NOX'], boston_df['INDUS'])
plt.xlabel('Nitric Oxide Concentration')
plt.ylabel('Proportion of Non-Retail Business Acres')
plt.title('Scatter Plot: NOX vs INDUS')
plt.show()


# # 5. Histogram of PTRATIO

# In[23]:


import matplotlib.pyplot as plt

# Create the histogram
plt.figure(figsize=(8, 6))
plt.hist(boston_df['PTRATIO'], bins=10, edgecolor='black')
plt.xlabel('Pupil-to-Teacher Ratio')
plt.ylabel('Frequency')
plt.title('Histogram of PTRATIO')
plt.show()


# # Task 5

# 1. Is there a significant difference in median value of houses bounded by the Charles river or not? (T-test for independent samples)

# In[24]:


import scipy.stats as stats

# Split the dataset into two groups based on Charles river (0: not bounded, 1: bounded)
charles_bounded = boston_df[boston_df['CHAS'] == 1]['MEDV']
charles_not_bounded = boston_df[boston_df['CHAS'] == 0]['MEDV']

# Perform the T-test for independent samples
t_stat, p_value = stats.ttest_ind(charles_bounded, charles_not_bounded)

# Define the significance level
alpha = 0.05

# Check if the p-value is less than the significance level
if p_value < alpha:
    print("There is a significant difference in median value of houses bounded by the Charles river.")
else:
    print("There is no significant difference in median value of houses bounded by the Charles river.")


# 2. Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)

# In[25]:


import scipy.stats as stats

# Create three groups based on the AGE variable
age_group_1 = boston_df[boston_df['AGE'] <= 35]['MEDV']
age_group_2 = boston_df[(boston_df['AGE'] > 35) & (boston_df['AGE'] <= 70)]['MEDV']
age_group_3 = boston_df[boston_df['AGE'] > 70]['MEDV']

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(age_group_1, age_group_2, age_group_3)

# Define the significance level
alpha = 0.05

# Check if the p-value is less than the significance level
if p_value < alpha:
    print("There is a significant difference in median values of houses for each proportion of owner-occupied units built prior to 1940.")
else:
    print("There is no significant difference in median values of houses for each proportion of owner-occupied units built prior to 1940.")


# 3. Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)

# In[26]:


import scipy.stats as stats

# Calculate the Pearson correlation coefficient and p-value
corr_coeff, p_value = stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])

# Define the significance level
alpha = 0.05

# Check if the p-value is greater than the significance level
if p_value >= alpha:
    print("We cannot conclude that there is a significant relationship between Nitric oxide concentrations and proportion of non-retail business acres per town.")
else:
    print("There is a significant relationship between Nitric oxide concentrations and proportion of non-retail business acres per town.")


# 4. What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes? (Regression analysis)
