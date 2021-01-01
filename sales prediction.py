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