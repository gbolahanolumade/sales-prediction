# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 23:26:54 2021

@author: GbolahanOlumade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("advertising.csv")

df.shape
df.info()
df.describe()


df(df.isnull())

df.isnull().sum()*100/df.shape[0]

df.isnull().sum()



plt.title("Outlier Search")
plt.xlabel("advert")
plt.ylabel("amount spent")
plt.boxplot(df)
plt.show()






fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(df['TV'], ax = axs[0])
plt2 = sns.boxplot(df['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(df['Radio'], ax = axs[2])
plt.tight_layout()


plt.scatter(df.TV,df.Sales)


sns.pairplot(df, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


plt.imshow(df, cmap='hot', interpolation='nearest')

# Let's see the correlation between different variables.
sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
plt.show()



X = df['TV'].ravel()
y = df['Sales'].ravel()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=100)

import statsmodels.api as sm

# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()

print(lr.summary())




plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


plt.scatter(X_train,res)
plt.show()

# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)


from sklearn.metrics import mean_squared_error, r2_score

np.sqrt(mean_squared_error(y_test, y_pred))

r_squared = r2_score(y_test, y_pred)
r_squared

plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()


X = df.iloc[:,0].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=100)


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)