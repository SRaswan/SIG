## . ./venv/bin/activate
'''
 The purpose of this program is to predict the stock price for the day
 based on the moving averages of stock prices for the past 3 days and 9 days.
'''

from sklearn.linear_model import LinearRegression 

import pandas as pd 
import numpy as np 

# for graphs
import matplotlib.pyplot as plt 
import seaborn 

# data from yahoo finance
import yfinance as yf


Df = yf.download('AAPL','2008-01-01','2022-10-1')
# only look at close columns
Df=Df[['Close']] 
# remove rows with missing values 
Df= Df.dropna() 
# plot the closing price 
Df.Close.plot(figsize=(10,5)) 
plt.ylabel("AAPL Prices")
plt.show()

# explanatory var uses moving averages from past 3 days AND past 9 days
Df['S_3'] = Df['Close'].shift(1).rolling(window=3).mean() 
Df['S_9']= Df['Close'].shift(1).rolling(window=9).mean() 

# remove rows with missing values
Df= Df.dropna() 
X = Df[['S_3','S_9']] 
X.head()

# dependent var is closing prices we are trying to predict
y = Df['Close']
y.head()

# split data between training data and testing data (first 80% is train)
t=.8 
t = int(t*len(Df)) 

# train dataset 
X_train = X[:t] 
y_train = y[:t]  
# test dataset 
X_test = X[t:] 
y_test = y[t:]

print(y_test.head(10))

# gets coefficent and constant for linear line of best fit
linear = LinearRegression().fit(X_train,y_train)

# predicts the price
predicted_price = linear.predict(X_test)  
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])  

# plots prediction
predicted_price.plot(figsize=(10,5))  
y_test.plot()  
plt.legend(['predicted_price','actual_price'])  
plt.ylabel("AAPL Price")  
plt.show()



# goodness of fit
r2_score = linear.score(X[t:],y[t:])*100
print(float("{0:.2f}".format(r2_score)))

