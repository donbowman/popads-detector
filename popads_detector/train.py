#!/usr/bin/env python3
#
# Copyright 2018 Don Bowman <db@donbowman.ca>
# Licensed under the Apache License, Version 2.0
#
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint

from popads_detector import data

chars = list(set(" abcdefghijklmnopqrstuvwxyz_-.01234567890"))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

max_model_len = 0
max_features = 0

def encode(domain):
    global max_model_len
    global char_indices

    domain = domain.ljust(max_model_len)
    encoded = []
    for i in range(0, max_model_len):
        encoded.append(char_indices[" "])
    for i, c in enumerate(domain):
        if (i==max_model_len):
            break
        if c in char_indices:
            encoded[i]=char_indices[c]
    return encoded

def create_model():
    global max_model_len
    global max_features
    max_features = len(chars)

    # dataset: {
    #  popads_domains: [ ... ]
    #  test_half_popads: [ ... ]
    #  train_half_popads: [ ... ]
    #  top_domains: [ ... ]
    #  test_top_domains: [ ... ]
    # }
    dataset = data.get_training_data()
    max_model_len = dataset['max_model_len']

    model=Sequential()
    model.add(Embedding(max_features, 128, input_length=dataset['max_model_len']))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model_json = model.to_json()
    with open("data/model.json", "w") as json_file:
        json_file.write(model_json)

    _xset = []
    _yset = []
    for i in range(0, len(dataset['train_half_popads'])):
        ds = encode(dataset['train_half_popads'][i])
        _xset.append(ds)
        _yset.append([1])

    for i in range(0, len(dataset['top_domains'])):
        _xset.append(encode(dataset['top_domains'][i]))
        _yset.append([0])

    xset = np.asarray(_xset)
    yset = np.asarray(_yset)
#    import pdb; pdb.set_trace()

    filepath = "data/trained.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model.fit(xset, yset, batch_size=900, callbacks=[checkpoint], validation_split=0.10, epochs=25)

