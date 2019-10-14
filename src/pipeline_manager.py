import pandas as pd
import numpy as np

from keras.layers import Dense, Embedding, Input, Conv2D, MaxPool2D, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, SpatialDropout1D, Reshape, Flatten, Concatenate
from keras.layers import GRU, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential, load_model


"""
    Notes: 
        1. Many ways to construct this pipeline logic
        2. Going with basic - function level of building
        3. Easy to extend to include different model architectures
        4. Here not trying to find best architecture, models are for demo only

"""
                   
def pipeline_model(word_index,
                  EMBEDDING_DIM,
                  embedding_matrix,
                  MAX_SEQUENCE_LENGTH,
                  MAX_NB_WORDS,
                  num_class,
                  pipeline_name):
    if(pipeline_name == 'baseline'):
            
        inp = Input(shape=(MAX_SEQUENCE_LENGTH, ))
        x = Embedding(len(word_index) + 1, EMBEDDING_DIM)(inp)
        x = Bidirectional(LSTM(50, return_sequences=True))(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(num_class, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)    
        
        return model
    
    if(pipeline_name == 'BiLSTM'):
            
        model = Sequential()
        model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=False))
        model.add(Bidirectional(LSTM(50, return_sequences=True)))
        
        model.add(GlobalMaxPool1D())
        
        model.add(Dropout(0.1))
        model.add(BatchNormalization())
        
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(0.1))
            
        model.add(Dense(num_class, activation="sigmoid"))
        
        return model
    
    if(pipeline_name == 'CNN2D'):
        filter_sizes = [1,2,3,5]
        num_filters = 32

        inp = Input(shape=(MAX_SEQUENCE_LENGTH, ))
        x = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
        x = SpatialDropout1D(0.4)(x)
        x = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(x)
        
        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], EMBEDDING_DIM), kernel_initializer='normal',
                                                                                        activation='elu')(x)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], EMBEDDING_DIM), kernel_initializer='normal',
                                                                                        activation='elu')(x)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], EMBEDDING_DIM), kernel_initializer='normal',
                                                                                        activation='elu')(x)
        conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], EMBEDDING_DIM), kernel_initializer='normal',
                                                                                        activation='elu')(x)
        
        maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1))(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1))(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1))(conv_2)
        maxpool_3 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[3] + 1, 1))(conv_3)
            
        z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
        z = Flatten()(z)
        z = Dropout(0.1)(z)
            
        outp = Dense(num_class, activation="sigmoid")(z)
        
        model = Model(inputs=inp, outputs=outp)        
    
        return model
    
    if(pipeline_name == 'GRUCNN'):

        inp = Input(shape = (MAX_SEQUENCE_LENGTH,))
        x = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights = [embedding_matrix], trainable = False)(inp)
        x1 = SpatialDropout1D(0.4)(x)
    
        x = Bidirectional(GRU(128, return_sequences = True))(x1)
        x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)    
    
        y = Bidirectional(LSTM(128, return_sequences = True))(x1)
        y = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(y)    
    
        avg_pool1 = GlobalAveragePooling1D()(x)
        max_pool1 = GlobalMaxPooling1D()(x)    
    
        avg_pool2 = GlobalAveragePooling1D()(y)
        max_pool2 = GlobalMaxPooling1D()(y)  
       
    
        x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])
    
        x = Dense(num_class, activation = "sigmoid")(x)
    
        model = Model(inputs = inp, outputs = x)   
            
        return model
    
    if(pipeline_name == 'CNNGRU'):
        
        inp = Input(shape=(MAX_SEQUENCE_LENGTH, ))
        x = Embedding(len(word_index) + 1, EMBEDDING_DIM)(inp)
        x = Dropout(0.4)(x)
        x = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = GRU(32)(x)
        x = Dense(16, activation="relu")(x)
        x = Dense(num_class, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
             
        return model


    
