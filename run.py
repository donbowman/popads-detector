#!/usr/bin/env python3
#
# Copyright 2018 Don Bowman <db@donbowman.ca>
# Licensed under the Apache License, Version 2.0
#

from  popads_detector import data
from  popads_detector import train

model = train.create_model()

def test_domain(model, domain, expected):
    r1 = train.model_lookup(model, domain)
    r2 = True
    if (expected and r1 < 0.8):
        r2 = False
    if ((not expected) and r1 > 0.3):
        r2 = False
    print("%s -->  %32s -> %4.2f (%s)" % (r2,domain, r1, expected))
    #print("Error: <<%s>> was expected %s, but got %4.2f" % (domain, expected, r1))
    return r2

dataset = data.get_training_data()

for domain in dataset['test_half_popads']:
    test_domain(model, domain, True)

for domain in dataset['top_test_domains']:
    test_domain(model, domain, False)
