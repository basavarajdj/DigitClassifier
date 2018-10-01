import keras
from keras.datasets import mnist
import numpy as np
from keras.layers import Convolution2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization


(trainx, trainy), (testx, testy) = mnist.load_data()

trainyy = keras.utils.to_categorical(trainy, num_classes=10)

model = Sequential()
model.add(BatchNormalization())
model.add(Convolution2D(filters=32, kernel_size=3, data_format='channels_last', activation='relu', input_shape=(28, 28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

trainxx = trainx.reshape(60000, 28, 28, 1)

model.fit(trainxx, trainyy, epochs=10, verbose=60000)

model.save('mnist_digit_convolution_w5e.h5')


#model = keras.models.load_model('mnist_digit_convolution_with_5_ephocs.h5')
#print(testx.shape)
testxx = testx.reshape(10000, 28, 28, 1)
testyy = keras.utils.to_categorical(testy, num_classes=10)
acc = model.evaluate(testxx, testyy)
#acc = keras.metrics.categorical_accuracy(testyy, pred)
#model.compile(optimizer='adam', loss='cateo')
#model.summary()

print(acc)
