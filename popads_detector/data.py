#!/usr/bin/env python3
#
# Copyright 2018 Don Bowman <db@donbowman.ca>
# Licensed under the Apache License, Version 2.0
#

import os
import tldextract
import random
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
    res = {}
    res['popads_domains'] = get_popads_domains()
    random.shuffle(res['popads_domains'])
    res['train_half_popads'] = res['popads_domains'][:int(len(res['popads_domains'])/2)]
    res['test_half_popads'] = res['popads_domains'][int(len(res['popads_domains'])/2):]

    top_domains = get_top_domains(len(res['popads_domains']))
    res['top_domains'] = top_domains[:int(len(res['popads_domains'])/2)]
    res['top_test_domains'] = top_domains[int(len(res['popads_domains'])/2):]

    random.shuffle(res['popads_domains'])

    res['max_model_len'] = len(max(
                      max(res['popads_domains'], key=len),
                      max(res['top_domains'], key=len)
                  ))

    return res
