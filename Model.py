
# coding: utf-8

# In[1]:

# Import all the initial libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import math
# get_ipython().run_line_magic('matplotlib', 'inline')

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Embedding, Flatten, Input, merge
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.constraints import non_neg

import requests
import scipy.sparse as sp
import argparse
import data_read as data
import metrics


# The ArgumentParser object will hold all the information necessary to parse the 
# command line into Python data types.
parser = argparse.ArgumentParser(description="Options")

# Add default value of the dataName to be 'ml-100k'
# for command line argument = -dataname 
parser.add_argument('-dataName', action='store', dest='dataName', default='ml-100k')

# number of training examples in one forward/backward pass
# higher the batchsize, more the memory you need 
# for command line argument = -batchSize
parser.add_argument('-batchSize', action='store', dest='batchSize', default=256, type=int)

# number of maximum training iterations
# for command line argument = -maxEpochs
parser.add_argument('-num_epochs', action='store', dest='num_epochs', default=10, type=int)

args = parser.parse_args()

# store the argument variable in the variable
dataName = args.dataName
batchSize = args.batchSize
num_epochs = args.num_epochs

# DATA EXPLORATION

# Load and transform data
# We're going to load the Movielens dataset

# create train and test data
train, test = data.get_movielens_data(dataName)
print(test)
# total number of users and items in the dataset
num_users, num_items = train.shape


# Creating the model

# latent vector dimensions for Factorization purposes.(used in GMF model)
latent_dim = 10
n_latent_factors_mf = latent_dim

# for neural network model (MLP, multi layer perceptron model variables).
n_latent_factors_user = 10
n_latent_factors_movie = 10


# There are three inputs: users, positive items, and negative items.
# In the triplet objective we try to make the positive item rank higher than the negative item for that user.
# Bayesian Personalized Ranking (Pairwise Objective function)
def bpr_triplet_loss(X):
    
    # item -> movie items
    # positive item latent vectors, negative item latent vectors, user latent vectors.
    positive_item_latent, negative_item_latent, user_latent = X

    # calculate BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss

# BUILD the model
def get_model(n_users, n_movies, n_latent_factors_mf, n_latent_factors_movie, n_latent_factors_user):
    
    # Input() is used to instantiate a Keras tensor.
    # For instance, if a, b and c are Keras tensors,
    # it becomes possible to do: model = Model(input=[a, b], output=c)
    
    # Positive and Negative item inputs.
    positive_item_input = Input((1, ), name='positive_item_input')
    negative_item_input = Input((1, ), name='negative_item_input')
    # user Input
    user_input = Input((1, ), name='user_input')

    
    # 'Embeddings': Turns positive integers (indexes) into dense vectors of fixed size. 
    
    # MF part
    # Shared embedding layer for positive and negative items
    item_embedding_layer = Embedding(n_movies + 1,
                                     n_latent_factors_mf,
                                     name='item_embedding',
                                     embeddings_constraint=non_neg(),
                                     input_length=1)
    
    # Flattens the input. Does not affect the batch size.
    positive_item_embedding = Flatten()(item_embedding_layer(positive_item_input))
    
    # Add dropouts for preventing overfitting.
    positive_item_embedding = keras.layers.Dropout(0.2)(positive_item_embedding)

    negative_item_embedding = Flatten()(item_embedding_layer(negative_item_input))
    negative_item_embedding = keras.layers.Dropout(0.2)(negative_item_embedding)

    # user embedding layers
    user_embedding = Flatten()(Embedding(n_users + 1,
                                         n_latent_factors_mf,
                                         name='user_embedding',
                                         embeddings_constraint=non_neg(),
                                         input_length=1)(user_input))
    user_embedding = keras.layers.Dropout(0.2)(user_embedding)
    
    # Pairwise ranking loss, 'BPR' loss (MF layer output)
    loss = merge([positive_item_embedding, negative_item_embedding, user_embedding],
                 mode=bpr_triplet_loss,
                 name='loss',
                 output_shape=(1, ))
    loss = keras.layers.Dropout(0.2)(loss)
    
    # MLP part
    # create user and item embeddings for MLP part
    
    # Shared embedding layer for positive and negative items
    item_embedding_layer = Embedding(n_movies + 1,
                                     n_latent_factors_movie,
                                     name='item_embedding_mlp',
                                     input_length=1)
    
    positive_item_embedding = Flatten()(item_embedding_layer(positive_item_input))
    positive_item_embedding = keras.layers.Dropout(0.2)(positive_item_embedding)

    negative_item_embedding = Flatten()(item_embedding_layer(negative_item_input))
    negative_item_embedding = keras.layers.Dropout(0.2)(negative_item_embedding)

    # User Embeddings
    user_vec_mlp = Flatten()(Embedding(n_users + 1,
                                     n_latent_factors_user,
                                     name='user_embedding_mlp',
                                     input_length=1)(user_input))
    user_vec_mlp = keras.layers.Dropout(0.2)(user_vec_mlp)
    
    concat = keras.layers.merge([positive_item_embedding,user_vec_mlp,negative_item_embedding],mode='concat',name='Concat-MLP')
    
    # Dense(): Just your regular densely-connected NN layer.
    # kernel is a weights matrix created by the layer, and
    # bias is a bias vector created by the layer (only applicable if use_bias is True).
    
    # layer 1 of the MLP model
    dense = keras.layers.Dense(64,name='FullyConnected_P',
                               use_bias=True,
                               kernel_initializer='lecun_uniform',
                               bias_initializer='zeros',
                               kernel_regularizer=l2(0.001),
                               bias_regularizer=l1(0.001),
                               activation='relu')(concat)
    
    # Normalize the activations of the previous layer at each batch,
    # i.e. applies a transformation that maintains the mean activation
    # close to 0 and the activation standard deviation close to 1.
    dropout_1 = keras.layers.BatchNormalization(name='Batch_P')(dense)
    #dropout_1 = keras.layers.Dropout(0.2,name='Dropout-1_P')(dense)
    
    # layer 2 of the MLP model
    dense_2 = keras.layers.Dense(32,
                                 name='FullyConnected-1_P', 
                                 use_bias=True,
                                 kernel_initializer='lecun_uniform',
                                 bias_initializer='zeros',
                                 kernel_regularizer=l2(0.001),
                                 bias_regularizer=l1(0.001),
                                 activation='relu')(dropout_1)
    dropout_2 = keras.layers.BatchNormalization(name='Batch-2')(dense_2)
    #dropout_2 = keras.layers.Dropout(0.2,name='Dropout-2_P')(dropout_2)
    
    # layer 3 of the MLP model
    dense_3 = keras.layers.Dense(16,name='FullyConnected-2_P',
                                 use_bias=True,
                                 kernel_initializer='lecun_uniform',
                                 bias_initializer='zeros',
                                 kernel_regularizer=l2(0.001),
                                 bias_regularizer=l1(0.001),
                                 activation='relu')(dropout_2)
    
    # last layer of the MLP
    pred_mlp = keras.layers.Dense(8, activation='relu',name='Activation_P')(dense_3)
    
    
    # concat the output of the 2 models (MF + MLP)
    combine_mlp_mf = keras.layers.merge([loss, pred_mlp], mode='concat',name='Concat-MF-MLP')
    
    # apply sigmoid activation function to obtain a prediction
    result = keras.layers.Dense(1, name='Prediction', activation='sigmoid')(combine_mlp_mf)
    
    # create the model object
    # tensor inputs of the model: positive_item_input, negative_item_input, user_input
    # output of the model: result
    model = keras.Model([positive_item_input, negative_item_input, user_input],
                        result)
    return model

# Build model
model = get_model(num_users, num_items, n_latent_factors_mf, n_latent_factors_movie, n_latent_factors_user)


# opt = keras.optimizers.Adam(lr =0.009)

# Compile the model by including optimization function and the loss function

# 'optimization function': adam (learning rate = default in keras model)
# 'loss function': binary crossentropy to keep the output of the predictions betweeen [0, 1]

model.compile(optimizer='adam',loss= 'binary_crossentropy')

# print the model summary, if needed
model.summary()

# list for storing the training loss
loss = []
best_hr, best_ndcg, best_iter = None, None, None

# Sanity check (if needed), AUC should be around 0.5
auc, ndcg_value, hr_value = metrics.full_auc(model, test)
print('Metric before training: ndcg %s, hr %s' % (ndcg_value, hr_value))

# iterate over the num_epochs (default = 10)
for epoch in range(num_epochs):

    print('Epoch %s' % epoch)

    # get Sample triplets from the training data (to be fed into the model)
    uid, pid, nid = data.get_triplets(train)

    X = {
        'user_input': uid,
        'positive_item_input': pid,
        'negative_item_input': nid
    }

    history = model.fit(X,
                      np.ones(len(uid)),
                      batch_size=batchSize,
                      epochs=1,
                      verbose=1,
                      shuffle=True)
    
    loss.append(history.history['loss'])
    
    # create triplets of (user, known positive item, randomly sampled negative item).
    # Prepare the test triplets (test data, used for 'predictions')
    # calculate the prediction values on the testing data created at the beginning of the code
    test_uid, test_pid, test_nid = data.get_triplets(test)
    
    X = {
        'user_input': test_uid,
        'positive_item_input': test_pid,
        'negative_item_input': test_nid
    }
    
    # calculate the metrics
    auc, ndcg_value, hr_value = metrics.full_auc(model, test)
    print('At epoch %s: ndcg %s, hr %s' % (epoch, ndcg_value, hr_value))
    
    if epoch == 0:
        best_hr, best_ndcg, best_iter = hr_value, ndcg_value, epoch
    else:
        if hr_value > best_hr or ndcg_value > best_ndcg:
            best_hr, best_ndcg, best_iter = hr_value, ndcg_value, epoch

# print('\nBest NDCG: {}, Best HR: {}, at epoch: {}\n'.format(best_ndcg, best_hr, best_iter))

# summarize history for loss  
# plot the training error or model loss
plt.plot(loss)
plt.title('MODEL LOSS')
plt.ylabel('LOSS')
plt.xlabel('EPOCHS')
plt.show()