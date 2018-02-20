#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:37:11 2018

@author: josem
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
#from datetime import timedelta

home = os.path.expanduser("~")
data_folder = os.path.join(home,"Escritorio/face_recognition/fc_data")

os.chdir(data_folder)

# HYPER-PARAMETERS
batch_size = 750
num_epochs = 10
num_channels = 3
learning_r = 0.0005
seed_of_tf = 1
img_size = 60
num_steps = 150
display_step = 10

# step 1 build pre-data in regular python
filenames = [f for f in os.listdir(data_folder)]
labels = []
for el in filenames:
    if "iniesta" in el:
        labels.append([1,0,0])
    elif "messi" in el:
        labels.append([0,1,0])
    else:
        labels.append([0,0,1])
        
print("\nImage names")
print(filenames[0:32])
print("\nCorresponding Labels")
print(labels[0:32])


## Build training and validation structure
data = pd.DataFrame(filenames,columns=["Names"])
data["Label"] = labels

np.random.seed(2)

T_indexes = np.random.choice(len(filenames),int(0.8*len(filenames)),replace=False)

T_data = data.iloc()[T_indexes]
V_data = data.drop(T_indexes)

T_filenames,T_labels = T_data["Names"].tolist(),T_data["Label"].tolist()
V_filenames,V_labels = V_data["Names"].tolist(),V_data["Label"].tolist()

with tf.device("/cpu:0"):
    tf.set_random_seed(seed_of_tf)
    # step 2: create a dataset returning slices of `filenames`
    T_dataset = tf.data.Dataset.from_tensor_slices((T_filenames,T_labels))
    V_dataset = tf.data.Dataset.from_tensor_slices((V_filenames,V_labels))
    
    # step 3: parse every image in the dataset using `map`
    def _parse_function(filename,label):
        #filename_print = tf.Print(filename,[filename])
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        return image,label
    
    T_dataset = T_dataset.map(_parse_function)
    V_dataset = V_dataset.map(_parse_function)

    #dataset = dataset.shuffle(buffer_size=10000)
    T_dataset = T_dataset.batch(batch_size)
    V_dataset = V_dataset.batch(1)

    T_dataset = T_dataset.repeat(num_epochs)
    
    # step 4: create iterator and final input tensor
    T_iterator = T_dataset.make_initializable_iterator()
    X,Y = T_iterator.get_next()
    
    V_iterator = V_dataset.make_initializable_iterator()
    X_V,Y_V = V_iterator.get_next()
    
    
    with tf.Session() as sess:
        sess.run(T_iterator.initializer)
    
        if True:
            checking_im,checking_l = sess.run([X,Y])
            print("\nVerifying tensor shapes")
            print(checking_im.shape)
            print(checking_l.shape)
            print("\nVerfiying image,label correspondence")
            imgplot = plt.imshow(checking_im[4,:,:,1])
            label = checking_l[4]
            plt.show()
            print(label)
    
    def conv_net(x, n_classes, reuse, is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):
            # CHALLENGE data input are images (60*60 pixels)
            # Reshape to match picture format [Height x Width x Channel]
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            x = tf.reshape(x, shape=[-1, img_size, img_size, 3])
    
            # Convolution Layer with 32 filters and a kernel size of 3
            conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    
            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

            # Convolution Layer with 128 filters and a kernel size of 3
            conv3 = tf.layers.conv2d(conv2, 128, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv3 = tf.layers.max_pooling2d(conv3, 2, 2)
    
            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv3)
    
            # Fully connected layer (in contrib folder for now)
            fc1 = tf.layers.batch_normalization(fc1)
            fc1 = tf.layers.dense(fc1, 30)#50
            # Apply Dropout (if is_training is False, dropout is not applied)
            #fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
    
            # Output layer, class prediction
            fc1 = tf.layers.batch_normalization(fc1)
            out = tf.layers.dense(fc1, n_classes)
            # Because 'softmax_cross_entropy_with_logits' already apply softmax,
            # we only apply softmax to testing network
            out = tf.nn.softmax(out) if not is_training else out
    
        return out 
    
    # Create a graph for training
    logits_train = conv_net(X, 3, reuse=False, is_training=True)
    # Create another graph for testing that reuse the same weights, but has
    # different behavior for 'dropout' (not applied).
    logits_Val = conv_net(X_V, 3, reuse=True, is_training=False)
    
    # Define loss and optimizer (with train logits, for dropout to take effect)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_train, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_r)
    train_op = optimizer.minimize(loss_op)
    
    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits_train, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred_V = tf.equal(tf.argmax(logits_Val, 1), tf.argmax(Y_V, 1))
    accuracy_V = tf.reduce_mean(tf.cast(correct_pred_V, tf.float32))
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    
    
    with tf.Session() as sess:
        # Run the initializer
        sess.run(T_iterator.initializer)
        sess.run(init)
        
        # Training cycle
        for step in range(1, num_steps + 1):
        
            try:
                # Run optimization
                sess.run(train_op)
            except tf.errors.OutOfRangeError:
                # Reload the iterator when it reaches the end of the dataset
                sess.run(T_iterator.initializer)
                sess.run(train_op)
        
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                # (note that this consume a new batch of data)
                loss, acc = sess.run([loss_op, accuracy])
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
        
        print("Optimization Finished!")
        
        # Validation cycle
        sess.run(V_iterator.initializer)
        acc_val = 0
        for step in range(len(V_data)):
            acc_val = acc_val + sess.run(accuracy_V)

        print("Validation finished with an accuracy of "+str(float(acc_val)/len(V_data)))
