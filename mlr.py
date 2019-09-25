import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX=LabelEncoder()
x[:, 3]=labelEncoderX.fit_transform(x[:, 3])
oneHotEncoder=OneHotEncoder(categorical_features=[3])
x=oneHotEncoder.fit_transform(x).toarray()

x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

#Backward Elimination
import statsmodels.formula.api as sm
x= np.append(arr=np.ones((50,1)).astype(int), values= x , axis=1)

x_opt = x[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
x_opt = x[:, [0,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
x_opt = x[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
x_opt = x[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
y_ols_pred= regressor_OLS.predict(x_test)