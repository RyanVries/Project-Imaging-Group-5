# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:07:17 2019

@author: s168003
"""
## Show 5 images of tissues without metastases and 5 images of tissues with metastases

names0 = ['', 
          '86de9372bb997cd6035632409db41d6e86af9956.jpg', 
          '86dfaa84c46733a5dcd66c0f9b4617f737c5101e.jpg', 
          '86e1d2e0b588aeb77481a59e19b08ff5913341e5.jpg',
          '86e4ab0936529351589ef3e218aad1918799fcc4.jpg',
          '86e6e628e6b881f4f8d00ead3f550815d1897fa0.jpg'] 
        #insert 5 image names including .jpg or .tif
names1 = ['', 
          '7518323a26fe0368a6245e4821ba8741ee0d566e.jpg', 
          '7520311b93c0f597e032006b3b5e4b34f229c5fb.jpg', 
          '7523674dbc4aeae1c52ede35e5cc03b7a5d3db12.jpg',
          '07530776f753d8d26c04eaaff29e9776fd89131b.jpg',
          '7534162eff11bd83be4bfae4034ee5423ddf695c.jpg'] 
        #insert 5 image names including .jpg or .tif

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

columns = 5
rows = 1

#without metastase
fig=plt.figure(figsize=(20, 4))
fig.suptitle('Without metastases', fontsize=14, fontweight='bold')
for i in range(1,6):
    img_name = 'train/0/' + names0[i]
    img = mpimg.imread(img_name)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis('off')

#with metastase
fig=plt.figure(figsize=(20, 4))
fig.suptitle('With metastases', fontsize=14, fontweight='bold')
for i in range(1,6):
    img_name = 'train/1/' + names1[i]
    img = mpimg.imread(img_name)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis('off')