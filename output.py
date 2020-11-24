# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:08:32 2019

@author: Oyelade
"""

runfile('C:/Transfer/Research Things/PostDoc/Coding/Paper1/tensorflow_keras_cnn_model.py', wdir='C:/Transfer/Research Things/PostDoc/Coding/Paper1')
Using TensorFlow backend.
Tensor("zero_padding2d/Pad:0", shape=(?, 1, 305, 305), dtype=float32)
WARNING:tensorflow:From C:\Users\Oyelade\Anaconda3\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
Tensor("conv1_1/3x3_s1/Relu:0", shape=(?, 32, 303, 303), dtype=float32)
Tensor("conv1_2/3x3_s1/Relu:0", shape=(?, 32, 301, 301), dtype=float32)
Tensor("zero_padding2d_1/Pad:0", shape=(?, 32, 303, 303), dtype=float32)
Tensor("flatten/Reshape:0", shape=(?, 65536), dtype=float32)
WARNING:tensorflow:From C:\Users\Oyelade\Anaconda3\lib\site-packages\tensorflow\python\keras\layers\core.py:143: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
Tensor("dropout/cond/Merge:0", shape=(?, 65536), dtype=float32)
Tensor("loss3/classifier/BiasAdd:0", shape=(?, 2), dtype=float32)
C:\Users\Oyelade\Anaconda3\lib\site-packages\tensorflow\python\keras\utils\conv_utils.py:225: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  return np.copy(kernel[slices])
WARNING:tensorflow:From C:\Users\Oyelade\Anaconda3\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Epoch 1/2
  11/1397 [..............................] - ETA: 899:55:25 - loss: 0.7805 - acc: 0.9432