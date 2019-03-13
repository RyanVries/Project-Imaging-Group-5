# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:26:56 2019

@author: s164616
"""

import os
import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.applications.inception_v3 import InceptionV3, preprocess_input


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val', 'train')
     valid_path = os.path.join(base_dir, 'train+val', 'valid')

     # instantiate data generators
     datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary')

     return train_gen, val_gen
 
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
    plt.plot(x, y[:,1], label='validation accuracy')
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()
    
    return

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)


input = Input(input_shape)

# get the pretrained model, cut out the top layer
pretrained = InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')

output = pretrained(input)
output = GlobalAveragePooling2D()(output)
output= BatchNormalization()(output)
output = Dropout(0.5)(output)
output = Dense(1, activation='sigmoid')(output)

model = Model(input, output)

# note the lower lr compared to the cnn example
model.compile(SGD(lr=0.001, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

# print a summary of the model on screen
model.summary()

# get the data generators
path='C:/Users/s164616/Documents/MATLAB/Project Imaging'
train_gen, val_gen = get_pcam_generators(path)

x=datetime.datetime.now()
extra='_'.join([str(x.year),str(x.month),str(x.day),str(x.hour),str(x.minute),str(x.second)])
        
# save the model and weights
model_name = 'InceptionV3_' + extra 
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model, note that we define "mini-epochs"
train_steps = train_gen.n//train_gen.batch_size//50
val_steps = val_gen.n//val_gen.batch_size//50

# since the model is trained for only 10 "mini-epochs", i.e. half of the data is
# not used during training
num_ep=10
history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=num_ep,
                    callbacks=callbacks_list)

predictions=model.predict_generator(val_gen,steps=val_steps*50)
y_pred = np.rint(predictions)
y_true = val_gen.classes


# ROC analysis
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)

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

