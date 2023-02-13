import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def get_returns(ticker, start_date):
    # Pull Historical Data
    data = yf.download(ticker, start=start_date)
    # Calculate Daily Returns
    data['Daily Return'] = data['Adj Close'].pct_change()
    return data.dropna()

#Rolling sharpe calculation (daily return pct divided by std of returns)
tnx = yf.Ticker('^TNX')
risk_free_rate = tnx.info.get('previousClose') * 0.01

# Northland
#data = get_returns('NPIFF', '2022-04-01')
#data = get_returns('NPI.TO', '2022-04-01')

# Ormat
data = get_returns('ORA', '2022-04-01')

data['sharpe'] = (data['Daily Return'] - risk_free_rate)/np.std(data['Daily Return'])
data.plot(kind='line', y = 'sharpe')
plt.show()