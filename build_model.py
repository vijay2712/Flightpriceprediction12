import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df=pd.read_csv('deploy_df')
df.drop('Unnamed: 0',axis=1,inplace=True)
x=df.drop('Price',axis=1)

y=df['Price']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=50)

from catboost import CatBoostRegressor
cat=CatBoostRegressor()
cat.fit(x_train,y_train)
y_predict=cat.predict(x_test)
print("Accuracy:",r2_score(y_test,y_predict))

import pickle
pickle.dump(cat,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(y_predict)