import cv2
import random
import numpy as np

# Can use dictionary instead of list
xs=[]
ys=[]

# Points to end of the last batch
train_batch_pointer=0
val_batch_pointer=0

# Read data.txt
with open('../../data/driving_dataset/data.txt') as f:
    for line in f:
        xs.append('data/driving_dataset/' + line.split()[0])
        # Paper by nvidia uses inverse of turning radius
        # but steering wheel angle is proportional to the inverse of turning radius
        # So the steering wheel in radians is used as output
        ys.append(float(line.split()[1])*3.14159265/180)                   # deg = rad * pi/180

# Get number of images
num_images = len(xs)

# Shuffle list of images
c=list(zip(xs,ys))
#random.shuffle(c)
xs,ys=zip(*c)

train_xs=xs[:int(len(xs)*0.8)]
train_ys=ys[:int(len(xs)*0.8)]

val_xs=xs[-int(len(xs)*0.2):]           #val_xs=xs[int(len(xs)*0.8):]
val_ys=ys[-int(len(xs)*0.2):]

num_train_images=len(train_xs)
num_val_images=len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out=[]
    y_out=[]
    for i in range(0,batch_size):
        x_out.append(cv2.resize(cv2.imread(train_xs[(train_batch_pointer+i)%num_train_images])[-150:],(200,66))/255.0)     # [-150]-> bottom 150 pxls of the image to train only half of the image of the input data to understand how much we have to rotate the steerig instantly
        y_out.append([train_ys[(train_batch_pointer+i)%num_train_images]])
        train_batch_pointer+=batch_size
    return x_out,y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out=[]
    y_out=[]
    for i in range(0,batch_size):
        x_out.append(cv2.resize(cv2.imread(val_xs[(val_batch_pointer+i)%num_val_images])[-150:],(200,66))/255.0)            # [-150] is the modification
        y_out.append(val_ys[(val_batch_pointer+i)%num_val_images])
        val_batch_pointer+=batch_size
    return x_out,y_out
