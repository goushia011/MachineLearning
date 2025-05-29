#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = pd.read_csv(r"C:\Users\Acer\Desktop\Python_Program\student-scores.csv")
print(dataset.shape)
FEATURES = dataset[['math_score','history_score','physics_score','chemistry_score','english_score','absence_days']]
X = FEATURES
y = dataset['weekly_self_study_hours']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('R2 Score: ',r2_score(y_test,y_pred))
plt.scatter(y_test,y_pred)
plt.plot(y_test,y_pred,color='red')
try:
    math = int(input("enter maths marks "))
    hist = int(input("enter hist marks "))
    phy = int(input("enter phy marks "))
    che = int(input("enter che marks "))
    eng = int(input("enter english marks "))

    predicted_hours = model.predict([[math, hist, phy, che, eng]])
    print(f'Estimated weekly self-study hours: ',predicted_hours)
except ValueError:
    print("Please enter valid integers for marks.")


# In[ ]:




