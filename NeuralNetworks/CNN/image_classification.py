import numpy
import keras


(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_train.info()
