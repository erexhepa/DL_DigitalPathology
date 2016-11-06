
# coding: utf-8

# In[33]:

#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import scipy.misc
import time
import argparse
import sys
import cv2

caffe_root = '/home/osboxes/caffe'


sys.path.insert(0, os.path.join(caffe_root , 'python'))
import caffe


# In[19]:

#this window size needs to be exactly the same size as that used to extract the patches from the matlab version
wsize = 32
hwsize= int(wsize/2)

BASE='/home/osboxes/docmac/TrainingTissueFinder/DL_DIGITALPATHOLOGY/DL_DigitalPathology/DL_tutorial_Code/case4_lymphocyte'
FOLD=2

#Locations of the necessary files are all assumed to be in subdirectoires of the base file
MODEL_DIR   = os.path.join(BASE,'models')
COMMON_DIR  = os.path.join('/home/osboxes/docmac/TrainingTissueFinder/DL_DIGITALPATHOLOGY/DL_DigitalPathology/DL_tutorial_Code/','common')

MODEL_FILE = os.path.join(COMMON_DIR,'deploy_train32.prototxt')
PRETRAINED = '%s/models/%d_caffenet_train_w32_iter_600000.caffemodel' % (BASE,FOLD)
IMAGE_DIR= '%s/images/' % BASE
OUTPUT_DIR= '%s/images/%d/' % (BASE,FOLD)


# In[3]:

OUTPUT_DIR


# In[16]:

#if our output directory doesn't exist, lets create it. each fold gets a numbered directory inside of the image directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# In[17]:

#load our mean file and reshape it accordingly
a = caffe.io.caffe_pb2.BlobProto()
file = open('%s/DB_train_w32_%d.binaryproto' % (MODEL_DIR,FOLD) ,'rb')
data = file.read()
a.ParseFromString(data)
means = a.data
means = np.asarray(means)
means = means.reshape(3, 32, 32)


# In[28]:

#make sure we use teh GPU otherwise things will take a very long time
#caffe.set_mode_gpu()
caffe.set_mode_cpu()
#load the model
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=means,
                       channel_swap=(2, 1, 0),
                       raw_scale=255,
                       image_dims=(32, 32))

#see which files we need to produce output for in this fold
#we look at the parent IDs in the test file and only compute those images
#as they've been "held out" from the training set
#files=open('%s/test_w32_parent_%d.txt'%(COMMON_DIR,FOLD),'rb')

start_time = time.time()
start_time_iter=0

#go into the image directory so we can use glob a bit easier
os.chdir(IMAGE_DIR)


# In[124]:

## One image test version

testDataDir = 'HER2/lymphocyte/data'
fname = os.path.join(BASE,testDataDir,'im15.tif')

newfname_class = fname.replace('.tif','_class.tif') #create the new files
newfname_prob = fname.replace('.tif','_prob.tif') #create the new files

outputimage = np.zeros(shape=(10, 10))
#first thing we do is save a file to let potential other workers know that this 
#file is being worked on and it should be skipped
scipy.misc.imsave(newfname_class, outputimage)

image = caffe.io.load_image(fname) #load our image
image = caffe.io.resize_image(image, [image.shape[0]*4,image.shape[1]*4]) #if you need to resize or crop, do it here
image = np.lib.pad(image, ((hwsize, hwsize), (hwsize, hwsize), (0, 0)), 'symmetric') #mirror the edges so that we can compute the full image
imageShape = image.shape
#image = cv2.resize(image,(imageShape[0]*4,imageShape[1]*4), interpolation = cv2.INTER_CUBIC)
imageFull = image
#image = image[0:140,0:150,0:3]

#plt.imshow(image)
#plt.show()
#plt.imshow(imageFull)
#plt.show()

sizeWind   = 64

outputimage_probs = np.zeros(shape=(sizeWind,sizeWind,3)) #make the output files where we'll store the data
outputimage_class = np.zeros(shape=(sizeWind,sizeWind))
final_outputimage_probs = np.zeros(shape=(imageFull.shape[0],imageFull.shape[1],3)) #make the output files where we'll store the data
final_outputimage_class = np.zeros(shape=(imageFull.shape[0],imageFull.shape[1]))

indexGridX = image.shape[0]/sizeWind
indexGridY = image.shape[1]/sizeWind

for ix in range(int(indexGridX)):
    for iy in range(int(indexGridY)):
        print (ix,iy)

        xStart  = (ix*sizeWind)
        xEnd    = ((ix+1)*sizeWind)
        yStart  = (iy * sizeWind)
        yEnd    = ((iy + 1) * sizeWind)
        image   = imageFull[xStart:xEnd,yStart:yEnd,:]

        for rowi in xrange(hwsize+1,image.shape[0]-hwsize):
            #print "%s\t (%.3f,%.3f)\t %d of %d" % (fname,time.time()-start_time,time.time()-start_time_iter,rowi,image.shape[0]-hwsize)
            start_time_iter = time.time()
            patches=[] #create a set of patches, oeprate on a per column basis

            for coli in xrange(hwsize+1,image.shape[1]-hwsize):
                patches.append(image[rowi-hwsize:rowi+hwsize, coli-hwsize:coli+hwsize,:])

            prediction = net.predict(patches) #predict the output
            pclass = prediction.argmax(axis=1) #get the argmax
            outputimage_probs[rowi,hwsize+1:image.shape[1]-hwsize,0:2]=prediction #save the results to our output images
            outputimage_class[rowi,hwsize+1:image.shape[1]-hwsize]=pclass

        #outputimage_probs = outputimage_probs[hwsize:-hwsize, hwsize:-hwsize, :] #remove the edge padding
        #outputimage_class = outputimage_class[hwsize:-hwsize, hwsize:-hwsize]

        final_outputimage_probs[xStart:xEnd,yStart:yEnd] = outputimage_probs
        final_outputimage_class[xStart:xEnd,yStart:yEnd] = outputimage_class

scipy.misc.imsave(newfname_prob,final_outputimage_probs) #save the files
scipy.misc.imsave(newfname_class,final_outputimage_class)


# In[127]:

512/64.0


# In[126]:

plt.imshow(final_outputimage_probs[:,:,1])
plt.show()


# In[ ]:

