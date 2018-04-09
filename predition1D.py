import numpy as np
np.random.seed(123)  # for reproducibility

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from numpy import genfromtxt
import numpy as np	
from matplotlib import pyplot as plt

# map code to number
code = {"G":0,"T":1,"A":2,"C":3}

#read csv test dataset
test_csv = genfromtxt('test.csv',dtype=None, delimiter=',',names=True)
testrows = test_csv.shape[0]
realtest_X = test_csv.reshape(testrows,1)['sequence']

# testing set decode and reshape
realtestX = np.zeros(shape=(testrows,14))
for index, item in enumerate(realtest_X):
  l = list(item[0])
  realtestX[index] = [code[chr(i)] for i in l[::-1]]
realtestX = realtestX.reshape(testrows, 14, 1)
print('Testing set shape: ', realtestX.shape)

realtestX = realtestX.astype('float32')
realtestX /= 3


#read csv training dataset
train_csv = genfromtxt('train.csv',dtype=(int,"|S14",int), delimiter=',',names=True)
train_rows = train_csv.shape[0]
train_X = train_csv.reshape(train_rows,1)['sequence']
trainY = train_csv.reshape(train_rows,1)['label']


# training set variable (2000,14)
trainX = np.zeros(shape=(train_rows,14))
for index, item in enumerate(train_X):
  l = list(item[0])
  trainX[index] = [code[chr(i)] for i in l[::-1]]

X_train = trainX.reshape(trainX.shape[0], 14, 1)
X_train = X_train.astype('float32')

X_test = X_train[900:1100,:,:] 

print('X_train shape: ', X_train.shape)

#normalize to [0,1]
X_train /= 3
X_test /= 3

Y_train = np_utils.to_categorical(trainY, 2)
Y_test = Y_train[900:1100,:] 

print('Y_train shape: ', Y_train.shape)

model = Sequential()
model.add(Convolution1D(32, 4, activation='relu', input_shape=(14,1)))
model.add(Convolution1D(64, 4, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(70, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


history = model.fit(X_train, Y_train, 
          batch_size=128, epochs=200, verbose=1)

plt.plot(history.history['acc'])
plt.show()

score = model.evaluate(X_test, Y_test, verbose=1)
print(score)
print(model.summary())

sample = model.predict(X_test, verbose=0)
print('Sample test result: ')
print(sample[0:100,1].sum(axis=0) + 100 -sample[101:200,1].sum(axis=0) ) 

output = model.predict(realtestX, verbose=1)

#print("Prediction result: ")
#print(output)

result = np.zeros(shape=(output.shape[0],2))

for index, item in enumerate(output):
  result[index]=[index, np.argmax(item)]

result = result.astype('int')

print("Predicted result saved as submission.csv")
np.savetxt("submission.csv", result, header="id,prediction", delimiter=",", comments='')
