import argparse
import librosa as lb
import numpy as np
from tqdm import tqdm
import os

import DynamicInput as dp

class dataprocessing:

    def preprocess(self, path, fs, hop_length, stem):
        try:
            Xtrain = np.load(path+'Xtrain.npy')
            Ytrain = np.load(path+'Ytrain.npy')
            print("Obtained Shape")
            print(Xtrain.shape, Ytrain.shape)
        except:
            print("Issue with the passed dataset path. Check name and data is in the specified path.\nCheck github for detailed explaination.")
            exit(1)

        if stem == 'vocals':
            s = 0
        elif stem == 'bass':
            s = 1
        elif stem == 'drums':
            s = 2
        elif stem == 'other':
            s = 3
    
        xtrain, ytrain = [], []
        for i in tqdm(range(Xtrain.shape[0])):
            
            bleed = Xtrain[i][s] 
            true = Ytrain[i][s]

            stft_bleed = lb.stft(bleed, hop_length = hop_length)
            stft_true = lb.stft(true, hop_length = hop_length)
            
            encode_bleed = dp.DynamicInput.encode(dp.DynamicInput, stft_bleed, 1, 1)
            encode_true = dp.DynamicInput.encode(dp.DynamicInput, stft_true, 1, 1)
            
            xtrain.append(encode_bleed)
            ytrain.append(encode_true)

        outpath = './processed_data/'
        os.makedirs(outpath, exist_ok=True)
        np.save(outpath+'Xtrain_processed.npy', np.vstack(xtrain))
        np.save(outpath+'Ytrain_processed.npy', np.vstack(ytrain))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CAE Data Preprocessing")

    # Add dataset path
    parser.add_argument("-dataset", "--path", type=str, help="Dataset Path (numpy files).")
    parser.add_argument("-fs", "--fs", type=int, default=22050, help="Sampling Frequency, defaults to 22.5khz")
    parser.add_argument("-hop_length", "--hop_length", type=int, default=2048, help="Hoplength, defaults to 2048")
    parser.add_argument("-stem", "--stem", type=str, help="specify vocals/bass/drums/other")

    args = parser.parse_args()

    dp = dataprocessing()
    dp.preprocess(args.path, args.fs, args.hop_length, args.stem)
    print('Preprocessing done.')
