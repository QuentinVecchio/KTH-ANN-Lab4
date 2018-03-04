import csv


def scenario_1():
    with open('../dataset/bindigit_trn.csv', 'rb') as csvfile:
        digitTrain = csv.reader(csvfile)
    print(len(digitTrain))
