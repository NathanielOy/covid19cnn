# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:33:43 2019

@author: Oyelade
"""

from __future__ import print_function
import imageio
#import matplolib as pyplot
#import matplotlib
'''
https://medium.com/@saedhussain/google-colaboratory-and-kaggle-datasets-b57a83eb6ef8
https://www.kaggle.com/general/51898
https://github.com/ieee8023/covid-chestxray-dataset
https://www.kaggle.com/general/74235

Datasets
https://www.kaggle.com/nih-chest-xrays/data
https://www.kaggle.com/bachrr/covid-chest-xray?select=metadata.csv
'''

#matplotlib.use('Agg')
#%matplotlib inline
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np
import os
import pandas as pd
from augmentation_op_2 import aug_train, aug_validation, aug_test, aug_train_mias3, aug_train_mias2, aug_train_mias1, aug_train_ddsm1, aug_train_ddsm2
from latest_input_fun2 import image_train_data, image_validation_data, image_test_data, getDataSize, getLabels, setLabels, get_validation_label, get_test_label, get_test_label2, get_test_label3
from pool_helper_2 import PoolHelper
from lrn import LRN
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
import pickle
from keras.models import Sequential, load_model

if keras.backend.backend() == 'tensorflow':
    from tensorflow.python.keras import backend as K
    import tensorflow as tf
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.disable_eager_execution()
    from tensorflow.python.keras.utils.conv_utils import convert_kernel
    from keras.backend.tensorflow_backend import set_session, clear_session, get_session
    #import multiprocessing as mp
    #mp.set_start_method('spawn', force=True)
    #tf.enable_eager_execution()
    
    # Create session config based on values of inter_op_parallelism_threads and
    # intra_op_parallelism_threads. Note that we default to having
    # allow_soft_placement = True, which is required for multi-GPU and not
    # harmful for other modes.
    NUM_PARALLEL_EXEC_UNITS=2
    config = tf.ConfigProto(
                        intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, 
                        inter_op_parallelism_threads=1, 
                        allow_soft_placement=True,
                        log_device_placement=True,
                        device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS})
    session = tf.Session(config=config)
    K.set_session(session)
    
    
    
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    #To not use GPU, a good solution is to not allow the environment to see any GPUs by setting the environmental variable CUDA_VISIBLE_DEVICES.
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    #tf.executing_eagerly()

SEED=42
IMG_WIDTH=220  #299  2560, 3328
IMG_HEIGHT=220  #299  2560, 3328
how = "normal"
batch_size=32
image_path_processed = "/content/drive/My Drive/CovidNet/dataset/processed"
covid_large_train=image_path_processed+"/train/"
d = covid_large_train
subfolders =[os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]

clasification_classes = []
for fld in subfolders:
    item=fld.split('/')
    clasification_classes.append(item[len(item)-1])
 
#print(clasification_classes)
print(len(clasification_classes))

'''
set the names of the classes to be classfied for use in the latest_input_fun.py file
'''
setLabels(clasification_classes)
num_classes = len(clasification_classes)
checkpoint='y' #y, n, model
epochs=5 #adam=12, sgd=17
isAug=False #False True
aug='sgd/'  #adam aug. sgd_aug, adam, sdg
optimizer='sgd' #sgd adam
l2_regulizer=0.0005 #0.0002 0.0005
filepath="/content/drive/My Drive/CovidNet/dataset/checkpoint/"



def create_cnn_model(weights_path=None):
    # creates our cnn model
    #filters which total weights is “n*m*k*l” (Here the input has l=32 feature maps as inputs, k=64 feature maps as outputs)
    #Then there is a term called bias for each feature map. So, the total number of parameters are “(n*m*l+1)*k”.
    '''
    PARAMETERS
    https://towardsdatascience.com/understanding-and-calculating-the-number-of-parameters-in-convolution-neural-networks-cnns-fc88790d530d
    https://medium.com/@shashikachamod4u/calculate-output-size-and-number-of-trainable-parameters-in-a-convolution-layer-1d64cae6c009
    https://medium.com/@iamvarman/how-to-calculate-the-number-of-parameters-in-the-cnn-5bd55364d7ca
    https://cs231n.github.io/convolutional-networks/
    '''
    
    input = Input(shape=(1, IMG_WIDTH, IMG_HEIGHT)) 
    input_pad = ZeroPadding2D(padding=(3, 3))(input)
    
    conv1_1_3x3_s1 = Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu', name='conv1_1/3x3_s1', kernel_regularizer=l2(l2_regulizer))(input_pad)
    conv1_2_3x3_s1 = Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu', name='conv1_2/3x3_s1', kernel_regularizer=l2(l2_regulizer))(conv1_1_3x3_s1) 
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_2_3x3_s1) 
    pool1_helper = PoolHelper()(conv1_zero_pad)
    pool1_2_2x2_s1 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same', name='pool1/2x2_s1')(pool1_helper)
    pool1_norm1 = LRN(name='pool1/norm1')(pool1_2_2x2_s1)
    
    conv2_1_3x3_reduce = Conv2D(64, (1,1), padding='same', activation='relu', name='conv2_1/3x3_reduce', kernel_regularizer=l2(l2_regulizer))(pool1_norm1)
    conv2_2_3x3 = Conv2D(64, (3,3), padding='same', activation='relu', name='conv2_2/3x3', kernel_regularizer=l2(l2_regulizer))(conv2_1_3x3_reduce)
    conv2_norm2 = LRN(name='conv2/norm2')(conv2_2_3x3)
    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
    pool2_helper = PoolHelper()(conv2_zero_pad)
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='pool2/3x3_s2')(pool2_helper)
    
    
    conv3_1_3x3_s1 = Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu', name='conv3_1/3x3_s1', kernel_regularizer=l2(l2_regulizer))(pool2_3x3_s2)
    conv3_2_3x3_s1 = Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu', name='conv3_2/3x3_s1', kernel_regularizer=l2(l2_regulizer))(conv3_1_3x3_s1)
    conv3_zero_pad = ZeroPadding2D(padding=(1, 1))(conv3_2_3x3_s1)
    pool3_helper = PoolHelper()(conv3_zero_pad)
    pool3_2_2x2_s1 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same', name='pool3/2x2_s1')(pool3_helper)
    pool3_norm1 = LRN(name='pool3/norm1')(pool3_2_2x2_s1)

    conv4_1_3x3_reduce = Conv2D(256, (1,1), padding='same', activation='relu', name='conv4_1/3x3_reduce', kernel_regularizer=l2(l2_regulizer))(pool3_norm1)
    conv4_2_3x3 = Conv2D(256, (3,3), padding='same', activation='relu', name='conv4_2/3x3', kernel_regularizer=l2(l2_regulizer))(conv4_1_3x3_reduce)
    conv4_norm2 = LRN(name='conv4/norm2')(conv4_2_3x3)
    conv4_zero_pad = ZeroPadding2D(padding=(1, 1))(conv4_norm2)
    pool4_helper = PoolHelper()(conv4_zero_pad)
    pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='pool4/3x3_s2')(pool4_helper)
    
     
    conv5_1_3x3_s1 = Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', name='conv5_1/3x3_s1', kernel_regularizer=l2(l2_regulizer))(pool4_3x3_s2)
    conv5_2_3x3_s1 = Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', name='conv5_2/3x3_s1', kernel_regularizer=l2(l2_regulizer))(conv5_1_3x3_s1)
    conv5_zero_pad = ZeroPadding2D(padding=(1, 1))(conv5_2_3x3_s1)
    pool5_helper = PoolHelper()(conv5_zero_pad)
    pool5_2_2x2_s1 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same', name='pool5/2x2_s1')(pool5_helper)
    pool5_norm1 = LRN(name='pool5/norm1')(pool5_2_2x2_s1)

    conv6_1_3x3_reduce = Conv2D(1024, (1,1), padding='same', activation='relu', name='conv6_1/3x3_reduce', kernel_regularizer=l2(l2_regulizer))(pool5_norm1)
    conv6_2_3x3 = Conv2D(1024, (3,3), padding='same', activation='relu', name='conv6_2/3x3', kernel_regularizer=l2(l2_regulizer))(conv6_1_3x3_reduce)
    conv6_norm2 = LRN(name='conv6/norm2')(conv6_2_3x3)
    conv6_zero_pad = ZeroPadding2D(padding=(1, 1))(conv6_norm2)
    pool6_helper = PoolHelper()(conv6_zero_pad)
    pool6_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='pool6/3x3_s2')(pool6_helper)
    
    
    pool7_2x2_s1 = AveragePooling2D(pool_size=(2,2), strides=(1,1), name='pool7/2x2_s1')(pool6_3x3_s2)
    
    loss_flat = Flatten()(pool7_2x2_s1)
    pool7_drop_2x2_s1 = Dropout(rate=0.5)(loss_flat)
    loss_classifier = Dense(num_classes, name='loss3/classifier', kernel_regularizer=l2(l2_regulizer))(pool7_drop_2x2_s1)
    loss_classifier_act = Activation('softmax', name='prob')(loss_classifier)

    mynet = Model(inputs=input, outputs=[loss_classifier_act])

    if weights_path:
        mynet.load_weights(weights_path)

    if keras.backend.backend() == 'tensorflow':
        # convert the convolutional kernels for tensorflow
        ops = []
        for layer in mynet.layers:
            if layer.__class__.__name__ == 'Conv2D':
                original_w = K.get_value(layer.kernel)
                converted_w = convert_kernel(original_w)
                ops.append(tf.assign(layer.kernel, converted_w).op)
        K.get_session().run(ops)

    return mynet
      
    
if __name__ == "__main__":

    filepath2=filepath+aug
    filepath+=aug+"_weights-{epoch:02d}-{loss:.4f}.hdf5"
     # Test pretrained model
    if checkpoint =='n':
        model = create_cnn_model()  #'googlenet_weights.h5'
    
    elif checkpoint =='model':
        custom_objects={'PoolHelper': PoolHelper(), 'LRN': LRN()}
        filepath2+="my_cnn_model.h5"
        model = tf.keras.models.load_model(filepath2, custom_objects=custom_objects)
    else:        
        filepath2+="my_cnn_weights.h5"
        model = create_cnn_model(filepath2)
        
    if optimizer == 'sgd':
        optim = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) #epsilon=1e-08, for Keras; epsilon=1e-08 for tensorflow; epsilon=1e-8 for Tocrch or MxNet 
    model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'],) 
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    filepath+=aug+"_weights-{epoch:02d}-{loss:.4f}.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') every improvement in accuracy
    #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min') every impromnet in loss
    #checkpointer = ModelCheckpoint(filepath='weights.hdf5', , monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1))
    checkpoint=ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10) # every epoch
    callbacks_list = [checkpoint, early_stopping]
    
    if not isAug: 
        train_dataset=image_train_data(K.get_session(), batch_size) 
        val_data=image_validation_data(K.get_session(), batch_size)  
        test_data=image_test_data(K.get_session(), batch_size)  
    else:
        base_dir="/content/drive/My Drive/CovidNet/dataset/processed/"
        trainaug=aug_train_ddsm1(batch_size)
        valaug=aug_validation(batch_size)
        testaug=aug_test(batch_size)
        
        train_dataset=trainaug.flow_from_directory(
                        directory=base_dir+"train/",
                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                        color_mode="grayscale",
                        batch_size=batch_size,
                        class_mode="sparse",
                        shuffle=True,
                        seed=SEED
                      )#train_fn_inputs(K.get_session(), epochs, batch_size, trainaug) 
        val_data=valaug.flow_from_directory(
                    directory=base_dir+"val/",
                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                    color_mode="grayscale",
                    batch_size=1,
                    class_mode="sparse",
                    shuffle=False,
                    seed=SEED
                )
        test_data=valaug.flow_from_directory(
                    directory=base_dir+"test/",
                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                    color_mode="grayscale",
                    batch_size=batch_size,
                    class_mode="sparse",
                    shuffle=True,
                    seed=SEED
                )
    
    train_records, val_records, test_records=getDataSize(1)
    TRAIN_STEPS_PER_EPOCH=int(train_records // batch_size)
    val_records=int(val_records)
    test_records=int(test_records)
    print(TRAIN_STEPS_PER_EPOCH)
    print(val_records//batch_size)
    print(test_records//batch_size)
    
    '''
    hist=model.fit_generator(train_dataset,
                   steps_per_epoch=int(train_records // batch_size),
                   epochs=epochs,
                   verbose = 1,
                   callbacks=callbacks_list,
                   validation_data=test_data, 
                   validation_steps=test_records//batch_size,
                   workers=0
              )
    
    mydir='/content/'  #'./'    #'./'  '/content/gdrive/My Drive/'
    print(hist.history)
    
    model_path1 = os.path.join(mydir, "my_cnn_weights.h5")
    model_path2 = os.path.join(mydir, "my_cnn_model.h5")
    model.save(model_path2)
    model.save_weights(model_path1)
    
    #model.summary()
    tf.keras.utils.plot_model(model, to_file=mydir+'archi_distortion_model2.png', show_shapes=True, show_layer_names=True)
    
    with open(mydir+'output.txt', 'w') as f:
        f.write(str(hist.history['loss']))
        f.write(str(hist.history['val_loss']))
        f.write(str(hist.history['acc']))
        f.write(str(hist.history))
    
    
    print("Training Loss: ", hist.history['loss'])
    print("Validation Loss: ", hist.history['val_loss'])
    print("Training Accuracy: ", hist.history['acc'])
    print("Training Accuracy: ", hist.history['val_acc'])
    '''
    
    score=model.evaluate_generator(val_data, 
                       steps=val_records//batch_size,
                       verbose = 1,
                       workers=0)
    print("Loss Val: ", score[0], "Accuracy Val: ", score[1])
    
    testX=val_data  #test_data, test_records
    step=val_records//batch_size
    
    '''
    if not isAug: 
        classes=get_test_label(K.get_session(),val_records) # get_test_label2(), get_test_label(K.get_session(),batch_size) get_validation_label()
    else:
        classes=testX.classes
    ''' 
    print('steps.....'+str(step))
    
    
    '''
    Y_pred = model.predict(testX, steps=step) 
    # reduce to 1d array
    #Y_pred = Y_pred[:, 0]
    yhat_classes = Y_pred.argmax(axis=1) #https://www.geeksforgeeks.org/numpy-argmax-python/
    diff_yhat_classes = np.argmax(Y_pred, axis = 1)
    print(Y_pred)
    print(Y_pred.shape)
    print(yhat_classes)
    print(diff_yhat_classes)
    print(classes)
    print(str(classes.shape)+'Confusion Matrix1'+str(yhat_classes.shape))
    print(confusion_matrix(classes, yhat_classes))
    print('Classification Report')
    print(classification_report(classes, yhat_classes))
    
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(classes, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(classes, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(classes, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(classes, yhat_classes)
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(classes, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(classes, Y_pred[:, 0])
    print('ROC AUC: %f' % auc)
    '''
    
    
    '''
    Begin: Special intervention
    '''
    base_dir="/content/drive/My Drive/CovidNet/dataset/processed/"
    valaug=aug_validation(batch_size)
    val_data_2=valaug.flow_from_directory(
                    directory=base_dir+"val/",
                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                    color_mode="grayscale",
                    batch_size=batch_size,
                    class_mode="sparse",
                    shuffle=False,
                    seed=SEED
                )
    Y_pred_2 = model.predict_generator(val_data_2,  step, verbose=1)
    yhat_classes_2 = Y_pred_2.argmax(axis=1) #https://www.geeksforgeeks.org/numpy-argmax-python/
    classes_2=val_data_2.classes
    print(Y_pred_2)
    print(Y_pred_2.shape)
    print(classes_2)
    print(str(len(classes_2))+'Confusion Matrix1'+str(yhat_classes_2.shape))
    print(confusion_matrix(classes_2, yhat_classes_2))
    cm = confusion_matrix(classes_2, yhat_classes_2)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Confusion matrix ')
    plt.colorbar()
    plt.show()
    
    print('Classification Report')
    print(classification_report(classes_2, yhat_classes_2))
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(classes_2, yhat_classes_2)
    print('Accuracy: %f' % accuracy)
    
    # precision tp / (tp + fp)
    precision = precision_score(classes_2, yhat_classes_2, average='micro')
    print('Precision micro: %f' % precision)
    precision = precision_score(classes_2, yhat_classes_2, average='macro')
    print('Precision macro: %f' % precision)
    precision = precision_score(classes_2, yhat_classes_2, average='weighted')
    print('Precision weighted: %f' % precision)
    precision = precision_score(classes_2, yhat_classes_2, average=None)
    print('Precision None: ' + str(precision))
    
    
    # recall: tp / (tp + fn)
    recall = recall_score(classes_2, yhat_classes_2, average='micro')
    print('Recall micro: %f' % recall)
    recall = recall_score(classes_2, yhat_classes_2, average='macro')
    print('Recall macro: %f' % recall)
    recall = recall_score(classes_2, yhat_classes_2, average='weighted')
    print('Recall weighted: %f' % recall)
    recall = recall_score(classes_2, yhat_classes_2, average=None)
    print('Recall None: ' + str(recall))
    
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(classes_2, yhat_classes_2, average='micro')
    print('F1 score micro: %f' % f1)
    f1 = f1_score(classes_2, yhat_classes_2, average='macro')
    print('F1 score macro: %f' % f1)
    f1 = f1_score(classes_2, yhat_classes_2, average='weighted')
    print('F1 score weighted: %f' % f1)
    f1 = f1_score(classes_2, yhat_classes_2, average=None)
    print('F1 score None: ' + str(f1))
    
    # kappa
    kappa = cohen_kappa_score(classes_2, yhat_classes_2)
    print('Cohens kappa: %f' % kappa)
    
    
    CM = confusion_matrix(classes_2, yhat_classes_2)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    specificity = TN / (TN+FP)
    sensitivity  = TP / (TP+FN)
    print('specificity3: %f' % specificity)
    print('sensitivity3: %f' % sensitivity)
    
    # ROC AUC
    try:
        auc = roc_auc_score(classes_2, Y_pred_2,  multi_class="ovr",average='macro')
        print('ROC AUC: %f' % auc)
        auc = roc_auc_score(classes_2, Y_pred_2,  multi_class="ovr",average='weighted')
        print('ROC AUC: %f' % auc)
    except ValueError:
        pass
    
    '''
    End: Special intervention
    '''
    
    
    #https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
    #https://machinelearningmastery.com/confusion-matrix-machine-learning/
    #https://www.tensorflow.org/guide/data
    #https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/
    #print(classification_report(classes, yhat_classes, target_names=clasification_classes))
    
    '''
    ValueError: Found input variables with inconsistent numbers of samples: [1197, 38304]
    Printing of graphs: Training, Validation and Testing
    
    N = epochs
    plt.style.use('seaborn-whitegrid')
    plt.title("Training/Validation Loss on Dataset ")
    plt.plot(np.arange(0, N),hist.history['loss'], label='training')
    plt.plot(np.arange(0, N),hist.history['val_loss'], label='validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch #")
    plt.legend(['training', 'validation'], loc="lower left")
    plt.show()
    
    plt.title("Training/Validation Accuracy on Dataset")
    plt.plot(np.arange(0, N),hist.history["acc"], label="train_accuracy")
    plt.plot(np.arange(0, N),hist.history["val_acc"], label="validation_accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch #")
    plt.legend(['training', 'validation'], loc="lower left")
    plt.show()
    
    
    plt.title("Evaluation Loss and Accuracy on Dataset")
    plt.plot(score[0], label='Eval-Loss')
    plt.plot(score[1], label="Eval Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(['validation loss', 'validation acc'], loc="lower left")
    plt.show();
    '''
    
    
    