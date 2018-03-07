#!/usr/bin/env python3
#
# Copyright 2018 Don Bowman <db@donbowman.ca>
# Licensed under the Apache License, Version 2.0
#

import os
import tldextract
import random
import pickle
from zipfile import ZipFile

import urllib.request

TOP_N_URL = 'http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip'
POPADS_URL = 'https://raw.githubusercontent.com/Yhonay/antipopads/master/hosts'

def get_top_domains(num, url=TOP_N_URL, filename='data/top-n.csv.zip'):
    if not os.path.isfile(filename):
        url = urllib.request.urlopen(url)
        zf = url.read()
        f = open(filename, 'wb')
        f.write(zf)
        f.close()

    res = {}
    with ZipFile(filename) as zf:
        with zf.open('top-1m.csv', 'r') as f:
            while (num > 0):
                zfb = f.readline().decode("utf-8")
                host = tldextract.extract(zfb.split(',')[1])
                #res.append(host.domain)
                if host.domain not in res:
                    res[host.domain.lower()] = 1
                    num -= 1
    return list(res.keys())

def get_popads_domains(url=POPADS_URL, filename='data/popads-hosts.txt'):
    if not os.path.isfile(filename):
        url = urllib.request.urlopen(url)
        zf = url.read()
        with open(filename, 'w') as f:
            f.write(zf.decode("utf-8"))

    res = {}
    with open(filename, 'r') as f:
        for zfb in f.readlines():
            host = tldextract.extract(zfb.split(' ')[1])
            if host.domain not in res:
                res[host.domain.lower()] = 1
    return list(res.keys())

def get_training_data():
    try:
        dataset = pickle.load(open("data/dataset.pkl","rb"))
    except FileNotFoundError:
        if not os.path.exists("data"):
            os.makedirs("data")

        dataset = {}
        dataset['popads_domains'] = get_popads_domains()
        random.shuffle(dataset['popads_domains'])
        dataset['train_half_popads'] = dataset['popads_domains'][:int(len(dataset['popads_domains'])/2)]
        dataset['test_half_popads'] = dataset['popads_domains'][int(len(dataset['popads_domains'])/2):]

        top_domains = get_top_domains(len(dataset['popads_domains']))
        dataset['top_domains'] = top_domains[:int(len(dataset['popads_domains'])/2)]
        dataset['top_test_domains'] = top_domains[int(len(dataset['popads_domains'])/2):]

        random.shuffle(dataset['popads_domains'])

        dataset['max_model_len'] = len(max(
                          max(dataset['popads_domains'], key=len),
                          max(dataset['top_domains'], key=len)
                      ))

        pickle.dump( dataset, open("data/dataset.pkl", "wb"))

    return dataset
