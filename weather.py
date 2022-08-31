__author__ = '3D'
__author__ = '3D'
import numpy
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# load the dataset
dataframe = pd.read_csv('data_2009_2016.csv', usecols=[1], skipfooter=3)
dataset = dataframe.values #numpy.ndarray
dataset = dataset.astype('float32')

numpy.random.seed(7)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.69)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)



# reshape into X=t and Y=t+1
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

from keras.callbacks import EarlyStopping
import time
# create and fit the LSTM network
model = Sequential() # New Instance of Model Object
model.add(LSTM(1024, input_shape=(1, look_back),return_sequences=True))
model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.3))

model.add(LSTM(512,return_sequences=True))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.3))


model.add(LSTM(256,return_sequences=True))
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.3))


model.add(LSTM(128))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.3))


model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='ADAM')
model.summary()
start = time.time()
hist =model.fit(trainX, trainY, epochs=1000, shuffle=True,batch_size=64, validation_data=(testX, testY),
                callbacks=[EarlyStopping(monitor='val_loss', patience=30)], verbose=1)

end = time.time()
# Training Phase
#model.summary()



print ("Model took %0.2f seconds to train"%(end - start))

logger=keras.callbacks.TensorBoard(log_dir='logs', write_graph=True)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
from sklearn.metrics import mean_absolute_error

print('Mean Absolute Error Test:', mean_absolute_error(testY[0], testPredict[:,0]))
print('Mean Squared Error Test:',np.sqrt(mean_squared_error(testY[0], testPredict[:,0])))
#print('Mean Absolute Percentage Error:',MAPError(testY[0], testPredict[:,0]))

print('Mean Absolute Error Train:', mean_absolute_error(trainY[0], trainPredict[:,0]))
print('Mean Squared Error Train:',np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0])))
#print('Mean Absolute Percentage Error:',MAPError(trainY[0], trainPredict[:,0]))


print('Mean Absolute Error Test:', mean_absolute_error(testY[0], testPredict[:,0]))
print('Mean Squared Error Test:',np.sqrt(mean_squared_error(testY[0], testPredict[:,0])))
#print('Mean Absolute Percentage Error:',MAPError(testY[0], testPredict[:,0]))
import seaborn as sns

sns.set_style('white')
sns.set_context("paper", font_scale=1.4)
plt.figure(figsize=(9,4))
import matplotlib as mpl
#mpl.rcParams['legend.frameon'] = 'True'
plt.plot(hist.history['loss'], color='#005ce6', linewidth=1, marker='d', markersize=7, label='Train Loss')
plt.plot(hist.history['val_loss'], color='#ffd11a',linewidth=1, marker='*',markersize=6, label='Test Loss')
plt.tick_params(left=False, labelleft=True) #remove ticks
plt.box(False)
plt.xlabel('Epochs')
plt.ylabel('Loss')
legend = plt.legend(loc='upper right',prop={'size': 16})
#legend.get_frame().set_facecolor('#8c8c8c')
plt.savefig('Fig15.4.png', dpi=400)


sns.set_style('white')
sns.set_context("paper", font_scale=1.4)
plt.figure(figsize=(9,4))
frame = legend.get_frame()
frame.set_color('white')

plt.plot(hist.history['loss'], color='#0000ff', linewidth=1, marker='d', markersize=7, label='Train Loss')
plt.plot(hist.history['val_loss'], color='#f2f2f2',linewidth=1, marker='*',markersize=5, label='Test Loss')
plt.tick_params(left=False, labelleft=True) #remove ticks
plt.box(False)
plt.xlabel('Epochs')
plt.ylabel('Loss')
legend = plt.legend(frameon = 1)
plt.legend(loc='best',prop={'size': 15})
frame.set_facecolor('blue')
frame.set_edgecolor('red')
plt.savefig('Fig15.4.png', dpi=400)

sns.set_style('white')
sns.set_context("paper", font_scale=1.5)
plt.figure(figsize=(8,4))

plt.plot(hist.history['loss'], marker='s', markersize=12, label='Train Loss')
plt.plot(hist.history['val_loss'], marker='*',markersize=12, label='Test Loss')
plt.tick_params(left=False, labelleft=True) #remove ticks
plt.box(False)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right');
plt.legend()
#plt.savefig('Fig8.png', dpi=300)


# plot baseline and predictions
# plt.figure(figsize=(22,5))
#
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()


trainPredict1=pd.DataFrame(trainPredict)
trainY1=pd.DataFrame(trainY)
trainY1=trainY1.transpose()

testPredict1=pd.DataFrame(testPredict)
testY1=pd.DataFrame(testY)
testY1=testY1.transpose()

sns.set_style('white')
sns.set_context("paper", font_scale=1.5)
plt.figure(figsize=(16,5))
plt.plot(testY1[:720], label = "Test Data")
plt.plot(testPredict1[:720], label = "Test Predict")
plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.xlabel('Time Index')
plt.ylabel('Energy Consumption')
plt.legend(loc='upper right')
plt.savefig('Fig8.png', dpi=500)
plt.show()


