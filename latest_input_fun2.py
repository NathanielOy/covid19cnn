# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:54:31 2019

@author: Oyelade
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
#tf.enable_eager_execution()
import os
import numpy as np
#import imageio
from PIL import Image
from math import floor
import pathlib
import PIL 
import IPython.display as display
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
from glob import glob
#import keras
import itertools
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#from keras_preprocessing.image import ImageDataGenerator

base_dir="/content/drive/My Drive/CovidNet/dataset"
image_path_processed = "/content/drive/My Drive/CovidNet/dataset/processed"

covid_large_images = image_path_processed+"/"
covid_large_images_labels = image_path_processed+"/Info.txt"
covid_large_train=image_path_processed+"/train/"
covid_large_val=image_path_processed+"/val/"
covid_large_test=image_path_processed+"/test/"
  
IMG_WIDTH=220 #299   1024 2560
IMG_HEIGHT=220 #299 1024  3328
AUTOTUNE=tf.data.experimental.AUTOTUNE
CLASS_NAMES=[] 

## Load the training data and return a list of the tfrecords file and the size of the dataset
## Multiple data sets have been created for this project, which one to be used can be set with the type argument
def getLabels():
    return CLASS_NAMES

def setLabels(clasification_classes):
    global CLASS_NAMES    
    CLASS_NAMES=clasification_classes
    print('Showing set classes...')
    print(CLASS_NAMES)
    
def getDataSize(dType):
    image_count=0
    val_count=0
    
    if dType ==1: #train data
        data=get_covid_image_training_data()
        val=get_covid_image_validation_data()
        test=get_covid_image_test_data()
        
        val_dir = pathlib.Path(val)
        val_count = len(list(val_dir.glob('*/*.png')))
        test_dir = pathlib.Path(test)
        test_count= len(list(test_dir.glob('*/*.png')))
        
    if dType ==2: #validation data
         data=get_covid_image_validation_data()
         
    data_dir = pathlib.Path(data)
    image_count = len(list(data_dir.glob('*/*.png')))
    print(image_count)
    print(val_count)
    print(test_count)
    #print([item for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    
    return image_count, val_count, test_count
   
def get_covid_image_training_data():
    return covid_large_train

def get_covid_image_validation_data():
    return covid_large_val

def get_covid_image_test_data():
    return covid_large_test

'getting images prepared as dataset format'
def get_label(file_path):
  parts = tf.strings.split(file_path, '/') # convert the path to a list of path components
  parts=tf.strings.split(parts[-1], '.')
  parts=tf.strings.split(parts[-2], '_')
  return parts[-1] == CLASS_NAMES  # The index last is the class-directory

def get_label2(file_path):
  parts = tf.strings.split(file_path, '/') # convert the path to a list of path components
  parts=tf.strings.split(parts[-1], '.')
  parts=tf.strings.split(parts[-2], '_')
  return parts[-1]   # The index last is the class-directory

def decode_img(img):
  img = tf.image.decode_png(img, channels=1) # convert the compressed string to a 3D uint8 tensor
  img = tf.image.convert_image_dtype(img, tf.float32) # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])  # resize the image to the desired size. #[1, IMG_WIDTH, IMG_HEIGHT]

def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path) # load the raw data from the file as a string
  img = decode_img(img)
  return img, label

def prepare_for_training(bs, ds, shuffle=True, cache=False, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.  
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  if shuffle:
      ds = ds.shuffle(buffer_size=shuffle_buffer_size).repeat(3).batch(bs).prefetch(1)
  else:
      ds = ds.repeat(3).batch(bs).prefetch(1)
  #plot_batch_sizes(ds)
  return ds

def plot_batch_sizes(ds):
  batch_sizes = [batch.shape[0] for batch in ds]
  plt.bar(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('Batch number')
  plt.ylabel('Batch size')
  
def image_train_data(sess, bs):
    train_image = get_covid_image_training_data()
    data_dir = pathlib.Path(train_image)
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.png')) 
    list_ds = list_ds.map(process_path)
    train_ds = prepare_for_training(bs, list_ds)
    train_ds=tf.compat.v1.data.make_initializable_iterator(train_ds)  
    image, label = train_ds.get_next()
    sess.run(train_ds.initializer)
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    image=image.eval(session=sess)
    label=label.eval(session=sess)
    '''
    from keras.utils import to_categorical
    y_binary = to_categorical(y_int)
    '''
    while True:
        yield image, label
    
def image_validation_data(sess, bs):
    val_files = get_covid_image_validation_data()
    data_dir = pathlib.Path(val_files)
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.png'))
    list_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = prepare_for_training(bs, list_ds, False)
    val_ds=tf.compat.v1.data.make_initializable_iterator(val_ds)  
    image, label = val_ds.get_next()
    sess.run(val_ds.initializer)
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    image=image.eval(session=sess)
    label=label.eval(session=sess)
    while True:
        yield image, label  
        
def image_test_data(sess, bs):
    test_files = get_covid_image_test_data()
    data_dir = pathlib.Path(test_files)
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.png')) 
    list_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    test_ds = prepare_for_training(bs, list_ds)
    test_ds=tf.compat.v1.data.make_initializable_iterator(test_ds)  
    image, label = test_ds.get_next()
    sess.run(test_ds.initializer)
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    image=image.eval(session=sess)
    label=label.eval(session=sess)
    while True:
        yield image, label
        
def get_validation_label(bs):
    val_files = get_covid_image_validation_data()
    val_dir = pathlib.Path(val_files)
    vals = (list(val_dir.glob('*/*.png')))
    labels=[]
    for sample in vals:
        label=get_label2(str(sample))
        labels.append(label)
        
    print(labels)
    return labels


'''
pred= model.predict_generator(validation_generator, nb_validation_samples // batch_size)
predicted_class_indices=np.argmax(pred,axis=1)
labels=(validation_generator.class_indices)
labels2=dict((v,k) for k,v in labels.items())
predictions=[labels2[k] for k in predicted_class_indices]

class_names = glob("*") # Reads all the folders in which images are present
class_names = sorted(class_names) # Sorting them
name_id_map = dict(zip(class_names, range(len(class_names))))
'''
def get_test_label3():
    test_files = get_covid_image_test_data()
    test_dir = pathlib.Path(test_files)
    class_names = test_dir.glob('*') # Reads all the folders in which images are present
    class_names = sorted(class_names) # Sorting them
    label_map = dict(zip(class_names, range(len(class_names))))
    '''
    The label_map variable is a dictionary like this
    {'class_14': 5, 'class_10': 1, 'class_11': 2, 'class_12': 3, 'class_13': 4, 'class_2': 6, 'class_3': 7, 'class_1': 0, 'class_6': 10, 'class_7': 11, 'class_4': 8, 'class_5': 9, 'class_8': 12, 'class_9': 13}
    which is equivalent to the values of a generator.class_indices
    '''
    return label_map, CLASS_NAMES

def get_test_label2():
    test_files = get_covid_image_test_data()
    test_dir = pathlib.Path(test_files)
    test = (list(test_dir.glob('*/*.png')))
    labels=[]
    for sample in test:
        label=get_label2(str(sample))
        idx=CLASS_NAMES.index(label)
        labels.append(idx)
        
    return labels

def get_test_label(sess, bs):
    test_files = get_covid_image_test_data()
    test_dir = pathlib.Path(test_files)
    list_ds = tf.data.Dataset.list_files(str(test_dir/'*/*.png')) 
    list_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    test_ds = prepare_for_training(bs, list_ds)
    test_ds=tf.compat.v1.data.make_initializable_iterator(test_ds)  
    image, label = test_ds.get_next()
    sess.run(test_ds.initializer)
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    image=image.eval(session=sess)
    label=label.eval(session=sess)
    return label
        
def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 1.0) # Make sure the image is still in [0, 1]
    return image, label

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    classes=cm_plot_labels = ['no_side_effects','had_side_effects']
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')