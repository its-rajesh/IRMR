import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, MaxPooling2D
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse


class tUNet:

    def model(self, inp):
        n, l, _ = inp.shape

        l1 = Conv2D(16, (1, n), activation='relu')(inp)
        l1 = MaxPooling2D(pool_size=(1, 9))(l1)

        l2 = Conv2D(32, (1, n), activation='relu')(l1)
        l2 = MaxPooling2D(pool_size=(1, 9))(l2)

        l3 = Conv2D(64, (1, n), activation='relu')(l2)
        l3 = MaxPooling2D(pool_size=(1, 9))(l3)

        l4 = Conv2D(128, (1, n), activation='relu')(l3)
        l4 = MaxPooling2D(pool_size=(1, 9))(l4)

        l5 = Conv2D(512, (1, n), activation='relu')(l4)
        l5 = MaxPooling2D(pool_size=(1, 9))(l5)

        flatten = Flatten()(l5)

        d1 = Dense(512, activation='relu')(flatten)
        d2 = Dense(128, activation='relu')(d1)
        d3 = Dense(64, activation='relu')(d2)
        d4 = Dense(32, activation='relu')(d3)
        d5 = Dense(n*n, activation='relu')(d4)
        out = Reshape((n, n))(d5)

        return out
    

    def model_train(self, X, A, batch_size, epochs):

        n = 4
        l = 220500 #(fs*10sec)
        inp = Input(shape=(n, l, 1))
        out = self.model(inp)
        model = Model(inp, out)

        opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=opt, loss='mse', metrics=['mse', 'mae'])
        md = model.fit(X, A, batch_size=batch_size, epochs=epochs, shuffle=True)
        model.save('tUNet.h5')

        plt.plot(md.history['mse'])
        plt.savefig('loss.png')

    def calc_interferenceMatrix(self, Xtrain, Ytrain):
        A_arr = []
        for i in tqdm(range(Xtrain.shape[0])):
            A_arr.append(Xtrain[i] @ np.linalg.pinv(Ytrain[i]))
        A_arr = np.array(A_arr)
        return A_arr



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="tUNet Train Script")

    # Add dataset path
    parser.add_argument("-dataset", "--path", type=str, default="./numpy_files/", help="Dataset Path (numpy files). default path is ./numpy_files/")
    parser.add_argument("-epochs", "--epochs", type=int, default=600, help="Number of epochs, defaults to 600.")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=64, help="Batch size, defaults to 64.")

    args = parser.parse_args()

    print("Loading Dataset")
    try:
        print(args.path)
        xtrain = np.load(args.path+'Xtrain.npy')
        ytrain = np.load(args.path+'Ytrain.npy')

        print('Data Shape Obtained:')
        print(xtrain.shape, ytrain.shape)

    except:
        print("Issue with the passed dataset path. Check name and data is in the specified path.\nCheck github for detailed explaination.")
        exit(1)

    print("Dataset loaded successfully")

    tunet = tUNet()
    print('Calculating Interference Matrix')
    A = tunet.calc_interferenceMatrix(xtrain, ytrain)
    print("Training..")
    tunet.model_train(xtrain, A, batch_size=args.batch_size, epochs=args.epochs)

    








