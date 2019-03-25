# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 12:02:01 2019

@author: s168003
"""

import os
import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Input, average ,concatenate, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model

# unused for now, to be used for ROC analysis
from sklearn.metrics import auc, roc_curve


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

def plot_tensorflow_log(path,num):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())

    training_accuracies =   event_acc.Scalars('acc')
    training_loss=event_acc.Scalars('loss')
    validation_accuracies = event_acc.Scalars('val_acc')
    validation_loss=event_acc.Scalars('val_loss')
    
    steps = num
    x = np.arange(steps)
    y = np.zeros([steps, 4])

    for i in range(steps):
        y[i, 0] = training_accuracies[i][2] # value
        y[i, 1] = validation_accuracies[i][2]
        y[i, 2] = training_loss[i][2]
        y[i, 3] = validation_loss[i][2]
        
    plt.plot(x, y[:,0], label='training accuracy')
    plt.plot(x, y[:,1], label='validation accuracy')

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend(loc='upper left', frameon=True)
    plt.show()
    
    plt.figure
    plt.plot(x, y[:,2], label='training loss')
    plt.plot(x, y[:,3], label='validation loss')
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()
    
    return


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     TRAIN_PATH = os.path.join(base_dir, 'train+val', 'train')
     VALID_PATH = os.path.join(base_dir, 'train+val', 'valid')

     RESCALING_FACTOR = 1./255
     
     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(TRAIN_PATH,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(VALID_PATH,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary',
                                             shuffle=False)
     
     return train_gen, val_gen
 


def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):

     # build the model
     model = Sequential()
     
     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Flatten())
     model.add(Dense(169, activation='relu'))
     model.add(Dense(169, activation='relu'))
     model.add(Dense(169, activation='relu'))
     model.add(Dense(169, activation='relu'))
     model.add(Dense(169, activation='relu'))
     model.add(Dense(169, activation='relu'))
     model.add(Dense(1, activation = 'sigmoid'))


     # compile the model
     model.compile(SGD(lr=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model



# get the model
model = get_model()


# get the data generators
path='C:/Users/s168003/.spyder-py3/assignments1-4'
train_gen, val_gen = get_pcam_generators(path)

x=datetime.datetime.now()
extra='_'.join([str(x.year),str(x.month),str(x.day),str(x.hour),str(x.minute),str(x.second)])
        
# save the model and weights
model_name = 'CamielNet' + extra 
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json) 




# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size
num_ep=10
history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=num_ep,
                    callbacks=callbacks_list)

predictions=model.predict_generator(val_gen,steps=val_steps)
#y_pred = np.rint(predictions)
y_true = val_gen.classes


# ROC analysis
fpr_keras,tpr_keras,_= roc_curve(y_true, predictions)
auc_keras=auc(fpr_keras,tpr_keras)

plt.figure()
plt.plot(fpr_keras, tpr_keras, color='darkorange', label='ROC curve (area = %0.2f)' % auc_keras)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

mypath=path+'/logs/'+model_name
file=os.listdir(mypath)
plot_tensorflow_log((mypath+'/'+file[0]),num_ep)

plot_model(model)
