#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
data = pd.read_csv(r"C:\Users\Acer\Desktop\Python_Program\Iris.csv")
X = data[['SepalLengthCm','PetalLengthCm','SepalWidthCm']]
y= data['PetalWidthCm']
model = LinearRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
print('R2 Score: ',r2_score(y_test,y_predict))


# In[ ]:





# In[ ]:




