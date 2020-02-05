
# coding: utf-8

# # Towards the Incorporation of Sex and Bile Acids in Alzheimer's Disease Prediction
# ## - Shadi Saee - 
# ## Exploratory Data Analysis

# ## Load Libraries and Data

# In[2]:

import random #neww
import os
import warnings
import statistics 
import scipy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.rcParams["font.family"] = "Times New Roman"  #global settings for font in plots

from math import sqrt
#from random import getrandbits
from statistics import mean, stdev, variance

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Code to suppress "future warnings"
from warnings import simplefilter
simplefilter(action ='ignore', category = FutureWarning)


# In[3]:


mcibl = pd.read_csv("Data/MCIBL.csv")
mcibl = mcibl.drop(mcibl.columns[[0]], axis = 1)


# ## Count and impute missing values based on variable mean

# In[4]:


na_values = mcibl.isnull().sum()[mcibl.isnull().sum() > 0]


# In[5]:


print(na_values)


# In[6]:


# Impute three missing values in ADAS13 column with the mean value
adas13_mean = mcibl["ADAS13"].mean()
mcibl.loc[:,"ADAS13"] = mcibl["ADAS13"].fillna(adas13_mean)


# ### Define subsets of data

# In[7]:


preds = mcibl.drop(["CONVERT", "RID"], axis = 1)
continuous_preds = preds.drop(["SEX", "APOE4"], axis = 1)
ba = mcibl.loc[:,["CA_CDCA", "DCA_CA","GLCA_CDCA", "GDCA_CA","GDCA_DCA", "TDCA_CA", "TLCA_CDCA", "TDCA_DCA"]]
y = mcibl.loc[:,"CONVERT"]


# ## Visual inspection of dataset: 

# ### Plot Probability Densitity plots of distribution of numerical variables

# In[8]:


# Plot distribution of demographic continuous variables: 
plt.figure(figsize = (7, 7))
plt.xlabel('Values', fontsize = 14)
plt.ylabel('Probability Density', fontsize = 14)
with sns.color_palette():
    p1=sns.kdeplot(mcibl["AGE"])
    p1=sns.kdeplot(mcibl["BMI"])
    p1=sns.kdeplot(mcibl["ADAS13"])
    p1=sns.kdeplot(mcibl["EDUCATION"]).set_title("Probability Density Distribution of Continous Demographic Variables", fontsize = 14)


# In[9]:


# Plot distribution of bile acid variables
plt.figure(figsize = (10, 10))
plt.title("Probability Density Distribution of Bile Acid Variables", fontsize = 14)
for col in ba:
    sns.kdeplot(mcibl[col])


# ### Histogram of MCI Converters vs. Non-Converters in Men vs. Women

# In[10]:


strat_count = pd.crosstab(mcibl.SEX, mcibl.CONVERT)


# In[11]:


strat_count.plot(kind = "bar", figsize = (7, 5))
plt.title("Progression from MCI to Alzheimer's Disease in Men vs. Women", fontsize = 14)
plt.xlabel('Sex', fontsize = 14)
plt.ylabel('Number of MCI Converters', fontsize = 14)
plt.legend(labels = ["MCI Non-Converters", "MCI Converters"], loc = "best", fontsize = 12)
plt.xticks(rotation = "horizontal", fontsize = 12)
#plt.savefig('Conversion by Gender')


# ### Plot proportion of MCI-Converters in Men vs. Women

# In[12]:


table = strat_count
table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (7, 5))

plt.title("Proportion of MCI Converters in Men and Women", fontsize = 14)
plt.xlabel('Sex', fontsize = 14)
plt.ylabel('Proportion of MCI Converters', fontsize = 14)
plt.legend(labels = ["MCI Non- Converters", "MCI Converters"])
plt.xticks(rotation = "horizontal", fontsize = 12)

#plt.savefig('Conversion Proportion by Gender')


# ### Plot Grouped Notched Boxplots of variables split by Conversion status and Sex

# In[13]:


for col in continuous_preds:
    plt.figure(figsize = (7, 7))
    sns.boxplot(x = "CONVERT", y = col, hue = "SEX", data = mcibl, notch  = True, 
                palette = "Set1").set_title( "Distribution of " + str(col) + ' Grouped by Sex and Conversion Status',
                                           fontsize = 14)
    plt.xlabel('Conversion Status', fontsize = 12)
    #plt.savefig("Distribution of " + str(col) + ' by Sex and MCI Conversion status')
plt.close()


# ## Quantitative inspection of dataset
# ### Calcuate odds ratio of progression to AD for men vs. women

# In[14]:


oddsratio, pvalue = sc.stats.fisher_exact(strat_count)
print("OddsR: ", oddsratio, "p-Value:", pvalue)


# ### Calculate group summary statistics stratified by conversion status and sex

# In[15]:


# Calculate group means and standard deviations grouped by conversion status and sex
strat_means = mcibl.drop("RID", axis = 1).groupby(["CONVERT","SEX"]).mean()
strat_means = pd.concat([strat_means], keys = ["Mean"], names = ["Mean"]).transpose()  #transpose and turn into dataframe

strat_std = mcibl.drop("RID", axis = 1).groupby(["CONVERT","SEX"]).std()
strat_std = pd.concat([strat_std], keys = ['Standard Dev.'], names = ["Standard Dev."]).transpose()


# In[16]:


# Calculate Effect size (Cohen's D) per column

# Group sample sizes
n_fem0 = strat_count[0]["Female"]
n_fem1 = strat_count[1]["Female"]
n_male0 = strat_count[0]["Male"]
n_male1 = strat_count[1]["Male"]

nonconvert_cohens_d = {}
convert_cohens_d = {}

for col in continuous_preds:
    col0_fem = mcibl[(mcibl.CONVERT == 0) & (mcibl.SEX == "Female")][col] #index data
    col1_fem = mcibl[(mcibl.CONVERT == 1) & (mcibl.SEX == "Female")][col]
    
    col0_male = mcibl[(mcibl.CONVERT == 0) & (mcibl.SEX == "Male")][col]
    col1_male = mcibl[(mcibl.CONVERT == 1) & (mcibl.SEX == "Male")][col]
    

    s0_fem, s1_fem = variance(col0_fem), variance(col1_fem)
    s0_male, s1_male = variance(col0_male), variance(col1_male)
    
    nonconvert_cohens_d[col] = (mean(col0_fem) - mean(col0_male)) / (sqrt(((n_fem0 - 1) * s0_fem + (n_male0 - 1) * 
                                                                           s0_male) / (n_fem0 + n_male0 - 2)))
    
    convert_cohens_d[col] = (mean(col1_fem) - mean(col1_male)) / (sqrt(((n_fem1 - 1) * s1_fem + (n_male1 - 1) * 
                                                                           s1_male) / (n_fem1 + n_male1 - 2)))
    

cohensd_df1 = pd.DataFrame([nonconvert_cohens_d, convert_cohens_d]).transpose()
cohensd_df1.columns = ["Effect Size Non-Converters", "Effect Size Converters"]

#Source: https://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python


# In[17]:


# Merge group means, standard deviation, and effect size data frames together 
frames = [strat_means, strat_std, cohensd_df1]
result = pd.concat(frames, axis = 1, sort = False).round(2) 

result.to_excel("Group Means.xlsx")
result.head()


# ### Calculate amount and proportion of APOE4 alleles greater than 1 per group

# In[18]:


apoe_fem_mci = mcibl[(mcibl.CONVERT == 0) & (mcibl.SEX == "Female")]["APOE4"]
apoe_male_mci = mcibl[(mcibl.CONVERT == 0) & (mcibl.SEX == "Male")]["APOE4"]
apoe_fem_ad = mcibl[(mcibl.CONVERT == 1) & (mcibl.SEX == "Female")]["APOE4"]
apoe_male_ad = mcibl[(mcibl.CONVERT == 1) & (mcibl.SEX == "Male")]["APOE4"]


def apoe_proportion(series):
    apoe_count = len(series[series >= 1])
    return apoe_count, apoe_count/len(series)


# In[19]:


for i in [apoe_fem_mci, apoe_male_mci, apoe_fem_ad, apoe_male_ad]:
    print(apoe_proportion(i))


# ### Plot Correlation Matrix of Mixed Dataset

# In[20]:


# Mixed dataset correlation plot
fig, ax = plt.subplots(figsize = (15,15))
sns.heatmap(continuous_preds.corr(), annot = True, ax = ax, cmap = "Blues") 
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)
#plt.savefig("Correlation Matrix Both Sexes")


# ### Inspect differences in variable correlations in men vs. women

# In[21]:


mci_female = mcibl[mcibl["SEX"] == "Female"]
mci_female = mci_female.drop(["SEX","RID"], axis = 1)
female_corr = mci_female.corr()

mci_male = mcibl[mcibl["SEX"] == "Male"]
mci_male = mci_male.drop(["SEX", "RID"], axis = 1)
male_corr = mci_male.corr()


# In[22]:


# Subtract male and female variable corrlations from each other to see if the correlations differ
corr_differences = male_corr-female_corr

fig, ax = plt.subplots(figsize = (15,15))
sns.heatmap(corr_differences, annot = True, ax = ax, cmap = "RdBu_r") 
plt.xticks(rotation = 45)
plt.savefig("Correlation Difference Both Sexes")


# ### Transform values of sex variable to numerical values

# In[73]:


le = preprocessing.LabelEncoder()

preds.loc[:,"SEX"] = le.fit_transform(preds["SEX"])
preds.head()


# ### Scale numerical variables

# In[74]:


scaler = StandardScaler()

sc_preds = scaler.fit_transform(continuous_preds)
sc_preds_df = pd.DataFrame(sc_preds, columns = continuous_preds.columns)


# In[80]:


print('Scaled mean: {}. Scaled standard deviation: {}'.format(sc_preds_df.values.mean(),sc_preds_df.values.std()))


# ## Principal Component Analysis

# In[76]:


continuous_preds.columns


# In[81]:


# Fit PCA algorithm 
pca = PCA()
pca.fit_transform(sc_preds_df)
explained_variance = pca.explained_variance_ratio_
princ_components = pca.components_

pca_df = pd.DataFrame(princ_components, columns = ['PC1', 'PC2', "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", 
                                             "PC9", "PC10", "PC11", "PC12"])
pca_df['Convert'] = y


# In[82]:


print(f'Percentage of explained variance per principal component: \n {explained_variance}')


# In[83]:


#Cumulative Variance explained
cum_var_exp = np.cumsum(np.round(explained_variance, decimals = 2))
cum_var_exp

#https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/


# In[95]:


plt.figure(figsize = (7,7))
plt.plot(cum_var_exp)
plt.xlabel('Number of Principal Components', fontsize = 14)
plt.ylabel('Cumulative Proportion of Explained Variance (%)', fontsize = 14) 
plt.title('Cumulative Proportion of Variance Explained by Principal Components', fontsize = 16)
plt.show()


# In[122]:


components_df = pd.DataFrame({'PC':['PC1','PC2','PC3','PC4',"PC5", "PC6", "PC7", "PC8","PC9", "PC10", "PC11", "PC12"],
                              'Explained Variance': pca.explained_variance_ratio_})

#f, ax = plt.subplots(figsize = (7,7))
ax = sns.barplot(x = 'PC', y = "Explained Variance", data = components_df, color = "navy")
ax.set_title('Proportion of Explained Variance by Principal Components')
fig = ax.get_figure()
plt.xlabel("Principal Component")
fig.savefig("Proportion of Explained Variance by Principal Components")
#components_df

