#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:37:40 2018

@author: ananthi
"""
from collections import Counter, OrderedDict
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from folder import DatasetFolder
from random import shuffle
from skimage import io
#import torchvision.models as models
#import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import SimpleITK as sitk
import torch.nn as nn
import numpy as np
import argparse
import shutil
import pylab
import torch
import cv2
#import os

import configuration
import data_loading
import preprocessing
import neural_network

DIRECTORY_ROOT: str = "/home/ananthi"

TRAINING_DATA_PATH: str = f"{DIRECTORY_ROOT}/hard-samples/"
TESTING_DATA_PATH: str = f"{DIRECTORY_ROOT}/test_GE/"
VALIDATION_DATA_PATH: str = f"{DIRECTORY_ROOT}/validate/"

MODEL_PATH: str = f"{DIRECTORY_ROOT}/model/myModel3b_best.pt"

parser = argparse.ArgumentParser(description='Microcalcification classifier')

parser.add_argument('--batch-size', type=int, default=40, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--test-batch-size', type=int, default=400, metavar='N',
                    help='input batch size for testing (default: 20)')
parser.add_argument('--validate-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for validating (default: 1000)')
parser.add_argument('--epochs', type=int, default=350, metavar='N',
                    help='number of epochs to train (default: 350)')
parser.add_argument('--epochs-validate', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 350)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                    help='SGD momentum (default: 0.8)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                    help='how many batches to wait before logging training status')

configuration_arguments: dict = configuration.parse_environment_args(parser)
config: configuration.Configuration = configuration.Configuration(**configuration_arguments)

torch.manual_seed(config.seed) # Sets the seed for generating random numbers. Returns a torch._C.Generator object
if config.cuda:
    torch.cuda.manual_seed(config.seed)

extra_data_loader_args: dict = {'num_workers': 18, 'pin_memory': True} if config.cuda else {}

# TODO: remove and return from orchestrator.run_training_testing()
avg_loss_1 = []
avg_loss_2 = []
best_auc = 0.0

unsharp_mask = preprocessing.UnsharpMaskCV2()
preprocessor = preprocessing.Preprocessor(unsharp_mask)
data_loader = data_loading.DataSetLoader(config, preprocessor, extra_data_loader_args)

training_data: data_loading.ImageFolder = data_loader.get_training_data_loader(TRAINING_DATA_PATH)
testing_data_loader: DataLoader = data_loader.get_testing_data_loader(TESTING_DATA_PATH)
hard_negative_testing_data_loader: DataLoader = data_loader.get_hard_negative_testing_data_loader(VALIDATION_DATA_PATH)

model: neural_network.Net = neural_network.initialize_neural_network(MODEL_PATH, config)

positive_samples = [(path, label) for (path, label) in training_data.samples if 'calc' in path]
negative_samples = [(path, label) for (path, label) in training_data.samples if 'negatives' in path]

optimizer = optim.SGD(
    model.parameters(), 
    lr=config.learning_rate, 
    momentum=config.momentum, 
    weight_decay=1e-5
)

orchestrator = neural_network.Orchestrator(model, optimizer, config)

# TODO: orchestrator.test_hard_negative() appends to avg_loss_2 as does orchestrator.test(), should these results be together?
# TODO: is this used at all?
avg_loss_2: list = orchestrator.test_hard_negative(hard_negative_testing_data_loader)

# Don't set a maximum if you want to keep the weights 'true': without samples to
# draw from, the sample will not be picked (if there are no 'difficult' samples,
# they can't be picked)

orchestrator.run_training_testing(training_data, positive_samples, negative_samples, testing_data_loader)

# If using hard negative testing data
#orchestrator.run_training_testing(training_data, positive_samples, negative_samples, hard_negative_testing_data_loader)

orchestrator.test_hard_negative(hard_negative_testing_data_loader)

train = plt.plot(avg_loss_1, color='blue', label='training set')
test = plt.plot(avg_loss_2, color='red', label='test set')
plt.legend()   
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.figure()
plt.plot(best_fpr,best_tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')

plt.figure()
plt.plot(best_fpr,best_tpr)
plt.xscale('log')
plt.xlabel('fpr_log')
plt.ylabel('tpr')

plt.figure()
plt.plot(best_fpr,best_tpr)
plt.xscale('log')
plt.xlim(10**-3,10**0)
plt.xlabel('partial_fpr_log')
plt.ylabel('tpr')

