import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.callbacks import TensorBoard


# load the dataset using the builtin Keras method
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# derive a validation set from the training set
# the original training set is split into 
# new training set (90%) and a validation set (10%)
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)



# the shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxH, where
# C is the channel dimension
X_train = np.reshape(X_train, (-1,28,28,1)) 
X_val = np.reshape(X_val, (-1,28,28,1))
X_test = np.reshape(X_test, (-1,28,28,1))


# convert the datatype to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')


# normalize our data values to the range [0,1]
X_train /= 255
X_val /= 255
X_test /= 255


# convert 1D class arrays to 4D class matrices, according to the groups from exercise 3
#First output node: "vertical digits": 1, 7
#Second output node: "loopy digits": 0, 6, 8, 9
#Third output node: "curly digits": 2, 5
#Fourth output node: "other": 3, 4
number_of_categories = 4
y_train_categories = np.zeros((len(y_train), number_of_categories), np.float32)
y_val_categories = np.zeros((len(y_val), number_of_categories), np.float32)
y_test_categories = np.zeros((len(y_test), number_of_categories), np.float32)

for i in range(len(y_train)):
    if (y_train[i] == 1) or (y_train[i] == 7):
        y_train_categories[i] = [1.,0.,0.,0.]
    elif (y_train[i] == 0) or (y_train[i] == 6) or (y_train[i] == 8) or (y_train[i] == 9): 
        y_train_categories[i] = [0.,1.,0.,0.]
    elif (y_train[i] == 2) or (y_train[i] == 5):
        y_train_categories[i] = [0.,0.,1.,0.]
    elif (y_train[i] == 3) or (y_train[i] == 4):
        y_train_categories[i] = [0.,0.,0.,1.]
    else:
        print("invalid output in y_train at position" + str(y_train[i]))

for i in range(len(y_val)):
    if (y_val[i] == 1) or (y_val[i] == 7):
        y_val_categories[i] = [1.,0.,0.,0.]
    elif (y_val[i] == 0) or (y_val[i] == 6) or (y_val[i] == 8) or (y_val[i] == 9): 
        y_val_categories[i] = [0.,1.,0.,0.]
    elif (y_val[i] == 2) or (y_val[i] == 5):
        y_val_categories[i] = [0.,0.,1.,0.]
    elif (y_val[i] == 3) or (y_val[i] == 4):
        y_val_categories[i] = [0.,0.,0.,1.]
    else:
        print("invalid output in y_val at position" + str(y_val[i]))
        
for i in range(len(y_test)):
    if (y_test[i] == 1) or (y_test[i] == 7):
        y_test_categories[i] = [1.,0.,0.,0.]
    elif (y_test[i] == 0) or (y_test[i] == 6) or (y_test[i] == 8) or (y_test[i] == 9): 
        y_test_categories[i] = [0.,1.,0.,0.]
    elif (y_test[i] == 2) or (y_test[i] == 5):
        y_test_categories[i] = [0.,0.,1.,0.]
    elif (y_test[i] == 3) or (y_test[i] == 4):
        y_test_categories[i] = [0.,0.,0.,1.]
    else:
        print("invalid output in y_test at position" + str(y_test[i]))

model = Sequential()
# flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
model.add(Flatten(input_shape=(28,28,1))) 
# fully connected layer with 64 neurons and ReLU nonlinearity
model.add(Dense(64, activation='relu'))
# output layer with 10 nodes (one for each class) and softmax nonlinearity
model.add(Dense(4, activation='softmax')) 

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# use this variable to name your model
model_name="my_first_model"

# create a way to monitor our model in Tensorboard
tensorboard = TensorBoard("logs/{}".format(model_name))

# train the model
model.fit(X_train, y_train_categories, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val_categories), callbacks=[tensorboard])

print("checkpoint 3")

score = model.evaluate(X_test, y_test_categories, verbose=0)


print("Loss: ",score[0])
print("Accuracy: ",score[1])
