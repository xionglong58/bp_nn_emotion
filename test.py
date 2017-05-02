#! /usr/bin/env python3
# -*-coding: utf-8-*-

__author__ = "Moonkie"


import numpy as np

t = np.random.randint(1,10,(3,4))
print(np.tanh(t))

def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

print(tanh(t))