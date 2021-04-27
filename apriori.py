#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys


# In[4]:



# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[7]:


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules


# In[8]:


df = pd.read_csv('./lumos1000.csv', header=None, dtype=None)


# In[9]:


df = df.replace(np.nan, '')


# In[10]:


records = []
for i in range(0, 1101):
    records.append([str(df.values[i,j]) for j in range(0, 7)])


# In[11]:


records = list(filter(None,[list(filter(None,l)) for l in records]))


# In[12]:


te = TransactionEncoder()


# In[13]:


te_ary = te.fit(records).transform(records)


# In[14]:


frame = pd.DataFrame(te_ary, columns=te.columns_)


# In[15]:


item = apriori(frame, min_support=0.0090, use_colnames=True)
print(item)
print(association_rules(item, metric="confidence", min_threshold=0.475))


# In[ ]:




