# -*- coding: utf-8 -*-
"""
@author: PraveenAdepu

Purpose: 
    1. Basic re-usable modularised pipeline
    2. Easy to extend any component
    3. Add additional model types without changing any component
    4. Single file "master.py" for both train, predict
    5. Pipeline controled using yaml config file
Value:
    1. Config file based execution
    2. Easy to test many ideas
    3. MVP level solution, qucik results
    4. Possibility to use in production subject to individual requirements
Data:
    1. dataset from kaggle competition
    2. https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
    3. dataset usage is for demo only, we can modify pipeline logic to any file format
       can easily modify for - regression, classification, multi classification
"""


import numpy as np

random_state = 201904
np.random.seed(201904)
np.random.RandomState(random_state)

import pandas as pd
import yaml as yaml

from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint
from keras.optimizers import SGD, Adam
from gensim.models import KeyedVectors
from keras.models import  load_model 

import os
import pickle

from src.generate_cv_folds import (StratifiedFolds)
from src.pipeline_manager import (pipeline_model)

print("Processing : Loading configuration file")
config = yaml.safe_load(open(r".\config\config.yaml"))

"""
assign config settings to local variables
"""

model_train = config['parameters']['model_train'] 
model_predict = config['parameters']['model_predict']

EMBEDDING_FILE = config['parameters']['EMBEDDING_FILE'] 
MAX_SEQUENCE_LENGTH = config['parameters']['MAX_SEQUENCE_LENGTH']
MAX_NB_WORDS = config['parameters']['MAX_NB_WORDS']
EMBEDDING_DIM = config['parameters']['EMBEDDING_DIM']
token_file = "./trained_models/"+config['parameters']['pipeline_name']+"_"+"tokenizer.pickle"
num_class = len(config['parameters']['target_feature_names'])
MODEL_WEIGHTS_FILE = "./trained_models/"+config['parameters']['pipeline_name']+"_"+config['parameters']['MODEL_WEIGHTS_FILE']

nb_epoch = config['parameters']['nb_epoch']
verbose = config['parameters']['verbose']
batch_size  = config['parameters']['batch_size']
patience = config['parameters']['patience']
learning_rate = config['parameters']['learning_rate']
cv_fold = config['parameters']['model_cross_validation_fold']  
pipeline_name = config['parameters']['pipeline_name'] 

train_filename = config['parameters']['train_filepath']
test_filename = config['parameters']['test_filepath']


def train():
    """
    Notes:
        1. read train dataset
        2. create cross validation folds
        3. train model on 4 folds, validate on 5th fold
        4. save model weights and tokenizer files
    """
    train = pd.read_csv(train_filename)
    cv_folds_columns = config['parameters']['target_feature_names']
    
    train['cv_folds_target'] = train[cv_folds_columns].sum(axis=1)
    
    train_cv = StratifiedFolds(df=train[['id','cv_folds_target']] , folds=config['parameters']['cross_validation_folds'], random_state=201904)
    del train_cv['cv_folds_target']
    del train['cv_folds_target']
    
    train = pd.merge(train, train_cv, on='id', how='left')
    
    
    list_sentences_train = train["comment_text"].fillna("missing").values
    list_classes = config['parameters']['target_feature_names']
    y = train[list_classes].values
    
   
    tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(list(list_sentences_train))
    
    """
    Notes:
        1. Saving tokenizer file to use at predict pipeline
    """
    
    with open(token_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)    
    X_train = sequence.pad_sequences(list_tokenized_train, maxlen=MAX_SEQUENCE_LENGTH)    
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))
    
    """
    Notes:
        1. Index word vectors
    """
    print('Indexing word vectors')
    
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
            binary=True)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))
    
    """
    Notes:
        1. Prepare embeddings
    """
    
    print('Preparing embedding matrix')
    
    nb_words = max(MAX_NB_WORDS, len(word_index))+1
    
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0)) #Null word embeddings: 96546
    
    """
    Notes : 
        1. Sample train/validation data
        2. We can loop this to train and predict for all folds
        3. Here training on 4 folds, validating on 5th fold
    """
    
    print('Fold ', cv_fold , ' Processing')
    
    trainindex = train[train['CVindices'] != cv_fold].index.tolist()
    valindex   = train[train['CVindices'] == cv_fold].index.tolist()
    
    X_build, y_build = X_train[trainindex], y[trainindex]
    X_valid, y_valid = X_train[valindex], y[valindex]   
       
    pred_cv = np.zeros([X_valid.shape[0],num_class])   
    
    """
    Notes:
        1. pipeline_name option is key to fetch corresponding model from all models
        2. On changing config, we can run entire pipeline without changing anything
        3. this is the whole idea of pipleline frameworks
    """
    model = pipeline_model(
                           word_index
                          ,EMBEDDING_DIM
                          ,embedding_matrix
                          ,MAX_SEQUENCE_LENGTH
                          ,MAX_NB_WORDS
                          ,num_class
                          ,pipeline_name = pipeline_name                     
                        ) 
    
    model.summary()
    """
    Notes:
        We can also do other things here, going with simple steps
        1. auto reduce lr
        2. csv logger
    """
    callbacks = [
    EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose),
    ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=verbose),
            ]
    
    optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_build, y_build, 
              validation_data=(X_valid, y_valid), 
              batch_size=batch_size, 
              epochs=nb_epoch,
              callbacks=callbacks,
              verbose=verbose
             )
    
    """
    Notes:
        1. Predict for cv fold
        2. We can write results back to database, we are not doing any saving
    """
    pred_cv = model.predict(X_valid, batch_size=batch_size, verbose=verbose)

def predict():
    """
    Notes:
        1. read test dataset
        2. create cross validation folds
        3. train model on 4 folds, validate on 5th fold
        4. save model weights and tokenizer files
    """
    print("reading test file")
    test = pd.read_csv(test_filename)
    list_classes = config['parameters']['target_feature_names']   
    list_sentences_test = test["comment_text"].fillna("missing").values 
    
    print("loading tokenizer pickle file")
    with open(token_file, 'rb') as handle:
        tokenizer = pickle.load(handle)        
    
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_test  = sequence.pad_sequences(list_tokenized_test, maxlen=MAX_SEQUENCE_LENGTH)
    
    print("loading model file")
    model = load_model(MODEL_WEIGHTS_FILE)
    
    print("predicting test file")
    pred_test = model.predict(X_test)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = list_classes
    pred_test["id"] = test.id.values
    
    """
    Notes:
        1. We can write results back to database or file
    """


if __name__ == '__main__': 
    """
    Notes:
        1. In production pipelines, we can train, predict using 
            config settings without changing anything
        2. Use the same pipeline to call saved model file to predict at real time
    """
    if model_train:
        print('model_train parameter : ', model_train , 'model training processing')
        train()
    else:
        print('model_train parameter : ', model_train , 'model training not processing')   
    
    if model_predict:
        print('model_predict parameter : ', model_predict , 'model predicting processing')
        predict()
    else:
        print('model_predict parameter : ', model_predict , 'model predicting not processing')   





