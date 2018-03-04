import csv
import numpy as np
from sklearn.neural_network import BernoulliRBM



def scenario_1():
    digitTrain = np.zeros((8000, 28*28))
    digitTest = np.zeros((2000, 28*28))
    targetTrain = np.zeros((8000, 1))
    targetTest = np.zeros((2000, 1))
    with open('../dataset/bindigit_trn.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        lines = []
        i = 0
        for row in reader:
            digitTrain[i] = np.array(row)
            i += 1
    with open('../dataset/bindigit_tst.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        lines = []
        i = 0
        for row in reader:
            digitTest[i] = np.array(row)
            i += 1
    with open('../dataset/targetdigit_trn.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        lines = []
        i = 0
        for row in reader:
            targetTrain[i] = np.array(row)
            i += 1
    with open('../dataset/targetdigit_tst.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        lines = []
        i = 0
        for row in reader:
            targetTest[i] = np.array(row)
            i += 1
    RBM = BernoulliRBM(n_components=100, learning_rate=0.1, n_iter=100, verbose=True)
    print("Start training")
    RBM.fit(digitTrain)
