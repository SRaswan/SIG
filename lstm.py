import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import matplotlib.dates as mdates
import copy

#symbol = 'NPI.TO'
#symbol = 'NPIFF'
symbol = 'ORA'

# todays_date = datetime.datetime.now().date()
# index = pd.date_range(todays_date, periods=360, freq='D')
# columns = ['Open']
# empty = pd.DataFrame(index=index, columns=columns)
# empty = empty.fillna(0)


training_complete = yf.download(symbol, '2016-01-01','2022-01-01')
training_processed = training_complete.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

training_scaled = scaler.fit_transform(training_processed)

features_set = []
labels = []
# for i in range(60, 1260):
for i in range(60, 1510):
    features_set.append(training_scaled[i-60:i, 0])
    labels.append(training_scaled[i, 0])

features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))

model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(features_set, labels, epochs = 1, batch_size = 32)

testing_complete =  yf.download(symbol, '2022-01-02','2022-11-15')
testing_processed = testing_complete.iloc[:, 1:2].values


#pdList = [training_complete['Open'], testing_complete['Open'], empty['Open']]
pdList = [training_complete['Open'], testing_complete['Open']]
total = pd.concat((pdList), axis=0)
#test_inputs = total[len(total) - len(testing_complete) - len(empty) - 60:].values
test_inputs = total[len(total) - len(testing_complete) - 60:].values
test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)
test_features = []
#for i in range(60, 310):
for i in range(60, 280):
    test_features.append(test_inputs[i-60:i, 0])

test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
predictions = model.predict(test_features)
predictions = scaler.inverse_transform(predictions)


# plt.figure(figsize=(10,6))
# plt.plot(testing_processed, color='blue', label='Actual Stock Price')
# plt.plot(predictions , color='red', label='Predicted Stock Price')

# plt.title('Stock Price Prediction')
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()

# future = copy.deepcopy(predictions)
for x in range(1):
    future = copy.deepcopy(predictions[-120:])
    test_inputs = future.reshape(-1,1)
    test_inputs = scaler.transform(test_inputs)
    test_features = []
    for i in range(60, 120):
        test_features.append(test_inputs[i-60:i, 0])
        # print(test_inputs[i-60:i, 0])
    test_features = np.array(test_features)
    #test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

    future = model.predict(test_features)
    future = scaler.inverse_transform(future)
    predictions = np.concatenate((predictions, future))


plt.figure(figsize=(10,6))
plt.plot(testing_processed, color='blue', label='Actual Stock Price')

plt.plot(predictions , color='red', label='Predicted Stock Price')

plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
