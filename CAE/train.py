from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, \
Flatten, BatchNormalization, Reshape
from tensorflow.keras.models import Model

import numpy as np
import argparse

class CAE:

    def model(self, inp):

        # Conv1 Layer
        x = Conv2D(filters = 32, kernel_size = (2, 1), activation='relu')(inp)
        x = BatchNormalization()(x)

        # Conv2 Layer
        x = Conv2D(filters = 64, kernel_size = (2, 1), activation='relu')(x)
        encode = BatchNormalization()(x)

        # Bottleneck
        x = Flatten()(encode)
        middle = Dense(100, activation='relu')(x)
        dense = Dense(encode.shape[1] * encode.shape[2] * encode.shape[3], activation='relu')(middle)
        reshape = Reshape((encode.shape[1], encode.shape[2], encode.shape[3]))(dense)

        # TConv2 Layer #
        x = Conv2DTranspose(filters = 32, kernel_size = (2, 1), activation='relu')(reshape)
        x = BatchNormalization()(x)

        # TConv1 Layer #
        x = Conv2DTranspose(filters = 1, kernel_size = (2, 1), activation='relu')(x)
        decode = BatchNormalization()(x)

        return decode
    

    def model_train(self, inp, X, Y, batch_size, epoch):

        decode = self.model(inp)
        di3_dca = Model(inp, decode)
        di3_dca.compile(optimizer="adam", loss="mse")
        print("Model compiled")

        print("CAE training..")
        history = di3_dca.fit(x=X, y=Y, epochs=epoch, batch_size=batch_size, shuffle=True, verbose=1)
        savemodelpath = './model.h5'
        di3_dca.save(savemodelpath)
        print('FINAL MODEL SAVED SUCCESSFULLY IN ({})'.format(savemodelpath))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CAE Train Script")

    # Add dataset path
    parser.add_argument("-dataset", "--path", type=str, default="./processed_data/", help="Dataset Path (Pre-processed). default path is ./processed_data/")
    parser.add_argument("-epochs", "--epochs", type=int, default=100, help="Number of epochs, defaults to 100.")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=64, help="Batch size, defaults to 64.")

    args = parser.parse_args()

    print("Loading Dataset")
    try:
        print(args.path)
        xtrain = np.load(args.path+'Xtrain_processed.npy')
        ytrain = np.load(args.path+'Ytrain_processed.npy')

        print('Data Shape Obtained:')
        print(xtrain.shape, ytrain.shape)

    except:
        print("Issue with the passed dataset path. Check name and data is in the specified path.\nCheck github for detailed explaination.")
        exit(1)

    print("Dataset loaded successfully")

    inp = Input(shape=(3075, 1, 1)) #Update here if you change specs.

    model = CAE()
    model.model_train(inp, xtrain, ytrain, args.batch_size, args.epochs)