#!/usr/bin/env python
# coding: utf-8

# The link for the Advanced regression dataset comes from:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques

# In[1]:


#Let's read the dataset 

import pandas as pd

df=pd.read_csv('train.csv')

#let's see the contents of dataset (will only view first 5 rows)

df.head()


# In[2]:


# Column Id is redundant , remove it!

df=df.drop(['Id'],axis=1)
df.head()


# 1. Now let's split the dataset based on numerical, categorical features
# 2. before doing that, let's separate the dependent/target variable i.e. SalePrice , let's name it as train_y
# 

# In[3]:


train_y=df['SalePrice']

df_num=df.select_dtypes(include=['int','float']).copy() #numerical features

df_cat=df.select_dtypes(include=['object']).copy() #categorical features


#lets drop the target variable from numerical features set

df_num.drop(['SalePrice'],axis=1)

#now you can see that SalePrice feature isn't showing up in df_num. ok cool.


# let's check number of numerical features and number of categorical features.!

# In[4]:


print("number of numerical features are: ",df_num.shape[1])
print("number of categorical features are: ",df_cat.shape[1])


# Let's check if we have any missing values in the columns of our dataset !

# In[5]:


def missing_values(df):
    nan_values=df_num.isna()
    nan_columns=nan_values.any() #It will tell if any of the columns have 

    # just uncomment next 2 lines i.e. remove the # from the lines and see if there are any missing values or not
    # print(nan_values)
    # print(nan_columns)

    #next 2 lines will let you see the name of the columns with missing values

    columns_with_nan = df_num.columns[nan_columns].tolist()
    if len(columns_with_nan)>=1:
        print("you have missing values")
    else:print("you don't have missing values")
    return columns_with_nan


# In[6]:


#Let's see if any of the numerical features have missing values or not
misslist=missing_values(df_num)
print(misslist)


# In[7]:


#Now let's deal with missing values..
#Lets fill up the missing positions by the average of the values of that particular features

nacol=misslist
for i in range(len(nacol)):
    m=df_num[nacol[i]].mean() # took the mean of feature 'LotFrontage' when i=0, 'ManVnrArea' when i=1 and so for 2.
    df_num[nacol[i]].fillna(m,inplace=True)
    
#we have filled missing values, but let's check to make sure. ok?

misslist=missing_values(df_num)


# ..
# Now, we shouldn't take all the features for regression, why? if we do use all features, it might affect negatively ! that is, reduce the model efficacy. 
# 
# Let's begin some feature engineering.!
# 
# 
# We will use P-test on features to choose good features. The p value will help us figure out the promising features. we will only take those features who give p values less than 0.05 while working with target SalePrice i.e. train_y. Let's do it. !
# 

# In[13]:


def feature_selection(df_num):
    from sklearn.datasets import load_boston
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm


    #df_num.drop('SalePrice',axis=1,inplace=True)

    X_1 = sm.add_constant(df_num)
    #Fitting sm.OLS model
    model = sm.OLS(train_y,X_1.astype(float)).fit()
    #print(model.pvalues)

    #Backward Elimination
    cols = list(df_num.columns)
    pmax = 1
    while (len(cols)>0):
        p= []
        X_1 = df_num[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(train_y,X_1.astype(float)).fit()
        p = pd.Series(model.pvalues.values[1:],index = cols)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    print("The selected features are:")
    print(selected_features_BE)


    df_n=df_num[selected_features_BE]

    return df_n


# **Section F**
# 
# In the above, we made a function, which will make a list of important features.
# Ok cool.
# 
# Now, we are gonna train our machine learning ...ummm. regression model..!! 
# 
# We will invoke scikit learn library to use their regression models !
# 
# And at first, we will use only numerical features to train and validate our model i.e. we will use df_num.

# In[ ]:





# In[15]:


from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE

#from sklearn import feature_selection
df_new=df_num
#df_new['LandContour']=df_cat['LandContour']
#df_new=df_new.drop(['SalePrice'],axis=1)
print(df_new.shape)

df_new=feature_selection(df_new)

#print(df_new.shape)



from sklearn.preprocessing import OrdinalEncoder


################
#X_new2 = SelectKBest(chi2, k=220).fit_transform(df_new,train_y)
#print(X_new2.shape)
#############
X_train = df_new[:-120]
X_test = df_new[-120:]

# Split the targets into training/testing sets
y_train = train_y[:-120]
y_test = train_y[-120:]


poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X_train)

X_test_ = poly.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
lg1 = LinearRegression()

lg1.fit(X_, y_train)
y_pred=lg1.predict(X_test_)
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
lg1.score(X_test_, y_test)


# So, we got score 86.12% by using numerical features and doing feature engineering on them. What if we didn't use the feature engineering?
# 
# Let's check it. just write df_new=df_num in line number 15 above and run it. 
# 
# what's the accuracy?
# It's -8.71, it's terrible! The MSE is 11 digit !
# 
# So, now it's obvious that we should do feature engineering.

# **Section G**
# <br>
# Now, let's add a categorical value to the train data,but before it, we have to convert it into numerical values. 

# In[ ]:


#We are taking the feature LandContour to convert it's string values to numerical values
#we will set, Bnk=1, Lvl=2, Low=3, HLS=4
cleanup_nums = {"LandContour": {"Lvl":2, "Bnk": 1, "HLS": 4, "Low":3}}
                
df_cat.replace(cleanup_nums, inplace=True)

#Now we will add it to the training set.

df_new['LandContour']=df_cat['LandContour']


# copy line 9 and paste in line 12 of section F.
# now run section F, and check & compare the score!
# 
# The score is 89.6% !
# ! Just adding a single categorical feature increased the score by 3% !! 
# 
# Now the question is, how I assigned the numerical values to the categorical values Lvl,Bnk,HLS,Low ? 
