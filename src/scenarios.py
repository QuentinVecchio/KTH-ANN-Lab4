import csv
import numpy as np
from sklearn.neural_network import BernoulliRBM
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt


def scenario_1():
    print("Loading data")
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
    print("Creating model")

    # this is the size of our encoded representations
    encoding_dim = 100

    # this is our input placeholder
    input_img = Input(shape=(784,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(784, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_output = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_output, decoder_layer(encoded_output))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    print(autoencoder.layers[2].get_weights()[0].shape)

    autoencoder.fit(digitTrain, digitTrain,
                epochs=100,
                batch_size=16,
                shuffle=True,
                verbose=2)

    # encode and decode some digits
    # note that we take them from the *test* set
    # encoded_imgs = encoder.predict(digitTest)
    # decoded_imgs = decoder.predict(encoded_imgs)
    #
    # n = 5  # how many digits we will display
    # plt.figure(figsize=(20, 4))
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(2, n, i + 1)
    #     plt.imshow(digitTest[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #
    #     # display reconstruction
    #     ax = plt.subplot(2, n, i + 1 + n)
    #     plt.imshow(decoded_imgs[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.show()


    n = 6  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(autoencoder.layers[2].get_weights()[0][i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1+n)
        plt.imshow(autoencoder.layers[2].get_weights()[0][i+n].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    plt.show()

def scenario_2():
    print("Loading data")
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
    print("Creating model")

    targetTest = np.true_divide(targetTest, 9)
    targetTrain = np.true_divide(targetTrain, 9)

    ########################
    #
    # input_img = Input(shape=(784,))
    # encoded = Dense(150, activation='relu')(input_img)
    # decoded = Dense(1, activation='sigmoid')(encoded)
    #
    # autoencoder = Model(input_img, decoded)

    # input_img = Input(shape=(784,))
    # encoded = Dense(150, activation='relu')(input_img)
    # encoded = Dense(75, activation='relu')(encoded)
    # decoded = Dense(1, activation='sigmoid')(encoded)

    input_img = Input(shape=(784,))
    encoded = Dense(150, activation='relu')(input_img)
    encoded = Dense(100, activation='relu')(encoded)
    encoded = Dense(50, activation='relu')(encoded)
    decoded = Dense(1, activation='sigmoid')(encoded)

    # input_img = Input(shape=(784,))
    # decoded = Dense(1, activation='sigmoid')(input_img)

    autoencoder = Model(input_img, decoded)

    ####################

    encoder = Model(input_img, encoded)

    encoded_output = Input(shape=(50,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_output, decoder_layer(encoded_output))

    # encoder = Model(input_img, decoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


    autoencoder.fit(digitTrain, targetTrain,
                epochs=100,
                batch_size=16,
                shuffle=True,
                verbose=2)

    # encode and decode some digits
    # note that we take them from the *test* set
    # encoded_imgs = encoder.predict(digitTest)
    # decoded_imgs = decoder.predict(encoded_imgs)

    decoded_imgs = encoder.predict(digitTest)

    nb = 0
    nb_correct = 0

    for i in range(len(decoded_imgs)):
        temp = decoded_imgs[i][0]
        temp = temp * 9
        temp = round(temp)
        if temp == 10:
            temp = 9
        if round(targetTest[i][0]*9) == temp:
            nb_correct += 1
        nb += 1

    print("Number of numbers: " + str(nb))
    print("Number of correct prediction: " + str(nb_correct))
    print("Ratio: " + str(nb_correct/nb))

    # n = 10  # how many digits we will display
    # plt.figure(figsize=(20, 4))
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(2, n, i + 1)
    #     plt.imshow(digitTest[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #
    #     # display reconstruction
    #     print(decoded_imgs[i][0]*9)
    #     # ax = plt.subplot(2, n, i + 1 + n)
    #     # plt.imshow(decoded_imgs[i].reshape(28, 28))
    #     # plt.gray()
    #     # ax.get_xaxis().set_visible(False)
    #     # ax.get_yaxis().set_visible(False)
    # plt.show()


    # n = 6  # how many digits we will display
    # plt.figure(figsize=(20, 4))
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(2, n, i + 1)
    #     plt.imshow(autoencoder.layers[2].get_weights()[0][i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #
    #     ax = plt.subplot(2, n, i + 1+n)
    #     plt.imshow(autoencoder.layers[2].get_weights()[0][i+n].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #
    #
    # plt.show()
