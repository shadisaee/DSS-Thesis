
# coding: utf-8

# ## Master's Thesis Data Science and Society - Shadi Saee
# ## Predicitve Machine Learning Models

# ### Load packages

# In[1]:


import os
import random
import warnings
import statistics
import scipy as sc
import numpy as np
import pandas as pd
import sklearn

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import RFE
from sklearn.metrics import recall_score, f1_score, confusion_matrix, precision_score
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, train_test_split


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

# Code to suppress "future warnings"
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# ### Load file

# In[2]:


mci = pd.read_csv("Data/MCIBL.csv")
mci = mci.drop(mci.columns[[0,1]], axis = 1)


# ### Change values of sex variable to numerical for processing in sklearn

# In[3]:


le = preprocessing.LabelEncoder()
le.fit(mci["SEX"])
print("Classes:", list(le.classes_))

mci["SEX"] = le.transform(mci["SEX"])
#1 = male, 0 = female 


# ## Create train/test split 

# In[4]:


# Stratify train and test set by the conversion and sex variables
train, test = train_test_split(mci, test_size = 0.2, random_state = 7, 
                               stratify = mci[['CONVERT', 'SEX']], shuffle = True)


# ### Check if sex and conversion variable are distributed with same proportions in train/test
# 

# In[5]:


prop_male_train = round(len(train.SEX[train.SEX == 1])/len(train), 2)
prop_male_test = round(len(test.SEX[test.SEX == 1])/len(test), 2)

prop_converter_train = round(len(test.CONVERT[test.CONVERT == 1])/len(test), 2)
prop_converter_test = round(len(train.CONVERT[train.CONVERT == 1])/len(train), 2)

print("Proportion of males in train set {} \nProportion of males in test {}\n".format(prop_male_train, prop_male_test))

print("Proportion of MCI Converters in train set {} \nProportion of MCI Converters in test set {}".format(prop_converter_train, prop_converter_test))


# ### Create predictor and outcome subsets
# 

# In[6]:


mci_full_train, mci_full_test = train.drop("CONVERT", axis = 1), test.drop("CONVERT", axis = 1)
mci_y_train, mci_y_test = train.loc[:,"CONVERT"], test.loc[:,"CONVERT"]


# In[7]:


# Counting Missing values in the dataframe
train_na = mci_full_train.isnull().sum()[mci_full_train.isnull().sum() > 0]
test_na = mci_full_test.isnull().sum()[mci_full_test.isnull().sum() > 0]


# In[8]:


print(train_na, test_na)


# In[9]:


adas_mean = mci_full_train["ADAS13"].mean()
mci_full_train.loc[:,"ADAS13"] = mci_full_train["ADAS13"].fillna(adas_mean)


# ### Subset male and female train/test sets 

# In[10]:


fem_train_index = mci_full_train["SEX"] == 0
fem_test_index = mci_full_test["SEX"] == 0

male_train_index = mci_full_train["SEX"] == 1
male_test_index = mci_full_test["SEX"] == 1


# In[11]:


# Function to subset male/female observations in dataset and delete sex variable  
def make_male(df, index):
    male = df[index]
    if isinstance(male, pd.DataFrame):  
        male = male.drop("SEX", axis = 1)
    return male

def make_female(df, index):
    female = df[index]
    if isinstance(female, pd.DataFrame):
        female = female.drop("SEX",axis = 1)
    return female 


# In[12]:


fem_full_train = make_female(mci_full_train, fem_train_index)
fem_full_test = make_female(mci_full_test, fem_test_index)

fem_y_train = make_female(mci_y_train, fem_train_index)
fem_y_test = make_female(mci_y_test, fem_test_index)


# In[13]:


male_full_train = make_male(mci_full_train, male_train_index)
male_full_test = make_male(mci_full_test, male_test_index)

male_y_train = make_male(mci_y_train, male_train_index)
male_y_test = make_male(mci_y_test, male_test_index)


# In[14]:


#print(male_full_train.shape,male_full_test.shape)


# In[15]:


#print(fem_full_train.shape, fem_full_test.shape)


# ### Scale continuous variables 

# In[21]:


continuous_preds = ["AGE", "EDUCATION","ADAS13","BMI", "CA_CDCA", "DCA_CA", "GLCA_CDCA", 
                    "GDCA_CA", "GDCA_DCA", "TDCA_CA", "TLCA_CDCA","TDCA_DCA"]


# In[22]:


scaler = StandardScaler()

# Function that scales continuous variables in train and test set
def scale_preds(train_df, test_df):
    train_df[continuous_preds] = scaler.fit_transform(train_df[continuous_preds])
    test_df[continuous_preds] = scaler.transform(test_df[continuous_preds])    
    return train_df, test_df


# In[23]:


sc_mci_full_train, sc_mci_full_test = scale_preds(mci_full_train, mci_full_test)
sc_fem_full_train, sc_fem_full_test = scale_preds(fem_full_train, fem_full_test)
sc_male_full_train, sc_male_full_test = scale_preds(male_full_train, male_full_test)


# ### Create subset of only demographic predictors for all three datasets

# In[24]:


# Create demographics predictor subset
demographics = ["AGE", "SEX", "EDUCATION", "APOE4", "ADAS13", "BMI"] 
sc_mci_demog_train, sc_mci_demog_test = sc_mci_full_train[demographics], sc_mci_full_test[demographics]

demographics.remove("SEX")

sc_fem_demog_train, sc_fem_demog_test = sc_fem_full_train[demographics], sc_fem_full_test[demographics]
sc_male_demog_train, sc_male_demog_test = sc_male_full_train[demographics], sc_male_full_test[demographics]


# ## Baseline - Majority Vote & Logistic Regression Model using only demographic predictors

# In[25]:


def confmatrix(y_test, y_pred):  
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred),
                      columns = ['pred_neg', 'pred_pos'], 
                      index = ['neg', 'pos'])
    return cm


# ### Majority Class Vote 

# In[26]:


dummy = DummyClassifier(strategy = 'most_frequent', random_state = 7)

def fit_dummy(x_train, y_train, x_test, y_test):
    dummy.fit(x_train, y_train)
    y_pred = dummy.predict(x_test)
    f1 = round(f1_score(y_test, y_pred),3)
    recall = round(recall_score(y_test,y_pred),3)
    precision = round(recall_score(y_test,y_pred),3)

    cm = confmatrix(y_test, y_pred)
    return cm, precision, recall, f1 


# In[27]:


mci_dummy_bl = fit_dummy(sc_mci_demog_train, mci_y_train, sc_mci_demog_test, mci_y_test)
fem_dummy_bl = fit_dummy(sc_fem_demog_train, fem_y_train, sc_fem_demog_test, fem_y_test)
male_dummy_bl = fit_dummy(sc_male_demog_train, male_y_train, sc_male_demog_test, male_y_test)


# In[28]:


print("MCI Dummy Baseline\n", mci_dummy_bl)
print("Female Dummy Baseline\n", fem_dummy_bl)
print("Male Dummy Baseline\n", male_dummy_bl)


# ### Define logistic regression function that returns estimate of test performance (F1 score) and confusion matrix

# In[29]:


logreg = LogisticRegression(max_iter = 500)

def fit_logreg(x_train, y_train, x_test, y_test):
    logreg.fit(x_train, y_train)
    y_pred_train = logreg.predict(x_train)
    #y_pred = logreg.predict(x_test)
    f1 = round(f1_score(y_train, y_pred_train),3)

    cm = confmatrix(y_train, y_pred_train)
    return f1, cm


# ### Fit logistic regresion model to all three datasets

# In[30]:


mci_logreg_bl = fit_logreg(sc_mci_demog_train, mci_y_train, sc_mci_demog_test, mci_y_test)
fem_logreg_bl = fit_logreg(sc_fem_demog_train, fem_y_train, sc_fem_demog_test, fem_y_test)
male_logreg_bl = fit_logreg(sc_male_demog_train, male_y_train, sc_male_demog_test, male_y_test)


# ### Calculate bootstrap F1 score on test data

# In[31]:


logreg = LogisticRegression(max_iter = 500)
ros = RandomOverSampler(random_state = 7, ratio = 1)  # ratio of minority class to majority class should be one


# In[32]:


def logreg_bootstrap_f1(x_train, x_test, y_train, y_test):
    f1 = []
    train_f1 = []
    precision = []
    recall = []
    
    np.random.seed(7)
    for i in range(500):
        # create bootsrap train set by randomly drawing with replacment from x_train 
        index = np.random.choice([True, False], size = x_train.shape[0])
        boot_x = x_train[index]
        boot_y = y_train[index]
        
        # within train, resample minority class with size of majority class to produce a balanced training set
        x_res, y_res = ros.fit_sample(boot_x, boot_y)        
        
        model = logreg.fit(x_res, y_res)
        y_pred = model.predict(x_test)
        y_pred_train = model.predict(boot_x)
        
        f1_test = f1_score(y_test, y_pred)
        f1.append(f1_test)
        
        f1_train = f1_score(boot_y, y_pred_train)
        train_f1.append(f1_train)
        
        precision_test = precision_score(y_test, y_pred)
        precision.append(precision_test)
        recall_test = recall_score(y_test, y_pred)
        recall.append(recall_test)
    
    
    cf = confmatrix(y_test, y_pred)
    mean_f1 = statistics.mean(f1)
    mean_f1_train = statistics.mean(train_f1)
    sample_std = np.std(f1, ddof = 1)   ## --> std error of f1
    mean_std_err = sample_std/(len(f1) ** 0.5)
    
    mean_recall = statistics.mean(recall)
    mean_precision = statistics.mean(precision)
    
    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(f1, p))
    lower = round(lower,4)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(f1, p))
    upper = round(upper, 4)
    ci = (lower*100, upper*100)
    
    return round(mean_f1_train,3), round(mean_f1,3), cf, ci, round(mean_precision,3), round(mean_recall,3) 

# Code CIs: https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/


# In[33]:


print("MCI Logistic Regression Baseline:")
logreg_bootstrap_f1(sc_mci_demog_train, sc_mci_demog_test, mci_y_train, mci_y_test)


# In[34]:


print("Female Logistic Regression Baseline:")
logreg_bootstrap_f1(sc_fem_demog_train, sc_fem_demog_test, fem_y_train, fem_y_test)


# In[35]:


print("Male Logistic Regression Baseline:")
logreg_bootstrap_f1(sc_male_demog_train, sc_male_demog_test, male_y_train, male_y_test)


# ## Grid Search: Model Selection and Hyperparameter Tuning
# ### Create a grid of different estimators and hyperparameter settings

# In[36]:


# Create a pipeline
pipe = Pipeline([("classifier", LogisticRegression())])

# Create dictionary with candidate learning algorithms and their hyperparameters
grid = [
                {"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2','l1', 'none'],
                 "classifier__solver" : ["liblinear", 'lbfgs'],
                 "classifier__max_iter": [100, 500, 1000],
                 "classifier__C": [0.001, 0.01, 0.1, 5]},  #regularization parameter
    
                {"classifier": [AdaBoostClassifier(random_state = 7)],
                 "classifier__n_estimators": [1, 10, 50],
                 "classifier__learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.3]},
    
                {"classifier": [RandomForestClassifier(random_state = 7)],
                 "classifier__n_estimators": [1, 10, 50, 100],
                 "classifier__max_depth": [5, 8, 15, 25, 30, None],
                 "classifier__criterion": ["entropy","gini"]},
        
                {"classifier": [KNeighborsClassifier()],
                 "classifier__n_neighbors": [2, 3, 5, 8, 10, 15, 20, 25, 50],
                 "classifier__weights": ["uniform","distance"]}]

# create a gridsearch of the pipeline, then fit the best model
gridsearch = GridSearchCV(pipe, grid , n_jobs = -1, cv = 10, verbose = 0, scoring = "f1") 


# Souce: https://towardsdatascience.com/hyper-parameter-tuning-and-model-selection-like-a-movie-star-a884b8ee8d68 


# ### Create dictionary containing train/test data for each group

# In[37]:


nested_dict = {"mci_demog": {"x_train": sc_mci_demog_train, "x_test": sc_mci_demog_test, 
                             "y_train": mci_y_train, "y_test": mci_y_test},
               
               "mci_full": {"x_train": sc_mci_full_train, "x_test": sc_mci_full_test, 
                             "y_train": mci_y_train, "y_test": mci_y_test,},
               
               "fem_demog": {"x_train": sc_fem_demog_train, "x_test": sc_fem_demog_test, 
                             "y_train": fem_y_train, "y_test": fem_y_test},
               
               "fem_full": {"x_train": sc_fem_full_train, "x_test": sc_fem_full_test, 
                            "y_train": fem_y_train, "y_test": fem_y_test},
               
               "male_demog": {"x_train": sc_male_demog_train, "x_test": sc_male_demog_test, 
                              "y_train": male_y_train, "y_test": male_y_test},
               
               "male_full": {"x_train": sc_male_full_train, "x_test": sc_male_full_test,
                             "y_train": male_y_train, "y_test": male_y_test}}


# ### Function that runs grid search and stores all information in the dictionary

# In[38]:


def run_search(dictionary):
    
    search = gridsearch.fit(dictionary["x_train"], dictionary["y_train"])
        
    dictionary["best_params"] = search.best_params_
    dictionary["cv_score"] = search.best_score_
    dictionary["classifier"] = search.best_params_["classifier"]
    
    return dictionary   


# ### Run grid search 

# In[39]:


#np.random.seed(7)
best_mci_demog = run_search(nested_dict["mci_demog"])
print(best_mci_demog["classifier"], round(best_mci_demog["cv_score"], 4))


# In[ ]:


#np.random.seed(10)
best_mci_fullpreds = run_search(nested_dict["mci_full"])
print(best_mci_fullpreds["classifier"], round(best_mci_fullpreds["cv_score"], 4))


# In[ ]:


#np.random.seed(7)
best_fem_demog = run_search(nested_dict["fem_demog"])
print(best_fem_demog["classifier"], round(best_fem_demog["cv_score"], 4))


# In[ ]:


#np.random.seed(10)
best_fem_fullpreds = run_search(nested_dict["fem_full"])
print(best_fem_fullpreds["classifier"], round(best_fem_fullpreds["cv_score"], 4))


# In[ ]:


#np.random.seed(7)
best_male_demog = run_search(nested_dict["male_demog"])
print(best_male_demog["classifier"], round(best_male_demog["cv_score"], 4))


# In[ ]:


#np.random.seed(7)
best_male_fullpreds = run_search(nested_dict["male_full"])
print(best_male_fullpreds["classifier"], round(best_male_fullpreds["cv_score"], 4))


# ### Create dataframe with CV results

# In[ ]:


cv_results_df = pd.DataFrame.from_dict(nested_dict).transpose()
cv_results_df = cv_results_df.drop(["x_test", "x_train", "y_test", "y_train"], axis = 1)
cv_results_df


# ## Estimating test error based on the mean boostrap test error

# In[ ]:


# Initialize random minority class oversampling function
ros = RandomOverSampler(random_state = 7, ratio = 1.0) #ratio of minority class to majority class = 1


# ### Function that calculates bootstrap test F1 score

# In[ ]:


def final_bootstrap_f1(n_dict):
    
    x_train = n_dict["x_train"]
    x_test = n_dict["x_test"]
    y_train = n_dict["y_train"]
    y_test = n_dict["y_test"]
    estimator = n_dict["classifier"]
    
    np.random.seed(7)
    f1 = []
    
    for i in range(500):
        # draw random sample with replacement from x_train with the same size as x_train
        index = np.random.choice([True,False], size = x_train.shape[0])        
        boot_x = x_train[index]
        boot_y = y_train[index]
        
        # within train, resample minority class with size of majority class to produce a balanced training set
        x_res, y_res = ros.fit_sample(boot_x, boot_y)        
        model = estimator.fit(x_res, y_res)

        y_pred = model.predict(x_test)
        score = f1_score(y_test, y_pred)
        f1.append(score)
    
    cf = confmatrix(y_test, y_pred)
    mean_f1 = statistics.mean(f1)
    sample_std = np.std(f1, ddof = 1)
    mean_std_err = sample_std/(len(f1) ** 0.5)
    
    n_dict["F1 Test"] = round(mean_f1, 4)
    n_dict["F1 Train"] = n_dict["cv_score"]
    
    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(f1, p))
    lower = round(lower,4)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(f1, p))
    upper = round(upper, 4)
    ci = (lower*100, upper*100)
    
    
    return  cf #, n_dict["F1 Test"],round(sample_std,4), ci


# Code CIs: https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/


# In[ ]:


for key in nested_dict.keys():
    print(key, final_bootstrap_f1(nested_dict[key]))


# In[ ]:


pd.DataFrame.from_dict(nested_dict).transpose()


# -----

# ## Apply mixed-sex classification model to female and male subsets

# ### Mixed sex model expects a sex variable, so I add a sex variable to the male and female subset encoding the sex
# 

# In[ ]:


n_male = male_full_test.shape[0]
n_female = fem_full_test.shape[0]

male_sex_values = n_male * [1]
female_sex_values = n_female * [0]

# Add 0/1 values to new sex variable in female and male subsets
sc_male_demog_test.loc[:,"SEX"] = male_sex_values
sc_male_full_test.loc[:,"SEX"] = male_sex_values

sc_fem_demog_test.loc[:,"SEX"] = female_sex_values
sc_fem_full_test.loc[:,"SEX"] = female_sex_values


# ### Adjust Bootstrap F1 function to take as input the training population and test population

# In[ ]:


def cross_sex_f1(train_pop, test_pop):
    
    x_train = train_pop["x_train"]
    y_train = train_pop["y_train"]
    
    x_test = test_pop["x_test"]
    y_test = test_pop["y_test"]
    estimator = train_pop["classifier"]
    
    np.random.seed(7)
    f1 = []
    
    for i in range(500):
        # draw random sample with replacement from x_train with the same size as x_train
        index = np.random.choice([True,False], size = x_train.shape[0])        
        boot_x = x_train[index]
        boot_y = y_train[index]
        
        # within train, resample minority class with size of majority class to produce a balanced training set
        x_res, y_res = ros.fit_sample(boot_x, boot_y)        
        model = estimator.fit(x_res, y_res)

        y_pred = model.predict(x_test)
        score = f1_score(y_test, y_pred)
        f1.append(score)
    
    cf = confmatrix(y_test, y_pred)
    mean_f1 = statistics.mean(f1)
    sample_std = np.std(f1, ddof = 1)
    mean_std_err = sample_std/(len(f1) ** 0.5)
    
    #n_dict["F1 Test"] = round(mean_f1, 4)
    #n_dict["F1 Train"] = n_dict["cv_score"]
    
    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(f1, p))
    lower = round(lower,4)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(f1, p))
    upper = round(upper, 4)
    ci = (lower*100, upper*100)
    
    return mean_f1, round(sample_std,4), ci, cf

# Code CIs: https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/


# ### Apply mixed sex models to male datasets

# In[ ]:


cross_sex_f1(nested_dict["mci_demog"], nested_dict["male_demog"])


# In[ ]:


cross_sex_f1(nested_dict["mci_full"], nested_dict["male_full"])


# ### Apply mixed-sex models to female datasets

# In[ ]:


cross_sex_f1(nested_dict["mci_demog"], nested_dict["fem_demog"])


# In[ ]:


cross_sex_f1(nested_dict["mci_full"], nested_dict["fem_full"])

