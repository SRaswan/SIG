import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import random
from scipy.stats import norm
from scipy.stats import stats

# Defining the Ticker
ticker = yf.Ticker('NPI.TO')

# Obtaining Historical Market Data
start_date = '2020-05-01'
end_date = '2022-11-04'
hist = ticker.history(start=start_date, end=end_date)
#print(hist.head())

# Pulling Closing Price Data
hist = hist[['Close']]
#print(hist)

# Plotting Price Data
hist['Close'].plot(title="Stock Price", ylabel= "Closing Price [$]", figsize=[10, 6])
plt.grid()

# Create Day Count, Price, and Change Lists
days = [i for i in range(1, len(hist['Close'])+1)]
price_orig = hist['Close'].tolist()
change = hist['Close'].pct_change().tolist()
change = change[1:]  # Removing the first term since it is NaN

# Statistics for Use in Model
mean = np.mean(change)
std_dev = np.std(change)
print('\nMean percent change: ' + str(round(mean*100, 2)) + '%')
print('Standard Deviation of percent change: '+str(round(std_dev*100, 2)) + '%')

# Simulation Number and Prediction Period
simulations = 50 # Change for more results
days_to_sim = 1*252 # Trading days in 1 year

# Initializing Figure for Simulation
fig = plt.figure(figsize=[10, 6])
plt.plot(days, price_orig)
plt.title("Monte Carlo Stock Prices [" + str(simulations) + " simulations]")
plt.xlabel("Trading Days After " + start_date)
plt.ylabel("Closing Price [$]")
plt.xlim([0, len(days)+days_to_sim])
plt.grid()

# Initializing Lists for Analysis
close_end = []
above_close = []
hpr = [] # holding period returns

# For Loop for Number of Simulations Desired
for i in range(simulations):
    num_days = [days[-1]]
    close_price = [hist.iloc[-1, 0]]
    last_price = close_price[-1] # before sim

    # For Loop for Number of Days to Predict
    for j in range(days_to_sim):
        num_days.append(num_days[-1]+1)
        perc_change = norm.ppf(random(), loc=mean, scale=std_dev)
        close_price.append(close_price[-1]*(1+perc_change)) # Markov chain

    if close_price[-1] > price_orig[-1]:
        above_close.append(1)
    else:
        above_close.append(0)

    close_end.append(close_price[-1])
    
    # calculate holding period return
    curr_hpr = (close_end[-1] - last_price)/last_price * 100
    hpr.append(curr_hpr)

    plt.plot(num_days, close_price)

# Average Closing Price and Probability of Increasing After 1 Year
average_closing_price = sum(close_end)/simulations
average_perc_change = (average_closing_price-
                       price_orig[-1])/price_orig[-1]
probability_of_increase = sum(above_close)/simulations
print('\nPredicted closing price after ' + str(simulations) + ' simulations: $' + str(round(average_closing_price, 2)))
print('Predicted percent increase after 1 year: ' + str(round(average_perc_change*100, 2)) + '%')
print('Probability of stock price increasing after 1 year: ' + str(round(probability_of_increase*100, 2)) + '%')

# Displaying the Monte Carlo Simulation Lines
plt.show()



sns.histplot(hpr, kde=True, bins=20, color='green')
plt.title('Holding Period Return Distribution')
plt.xlabel('HPR %')
plt.axvline(np.percentile(hpr, 5), color='r', linestyle='dashed', linewidth=2)
plt.axvline(np.percentile(hpr, 95), color='r', linestyle='dashed', linewidth=2)
plt.show()


