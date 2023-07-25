import os
import cmath
import argparse
import numpy as np
import librosa as lb
from tqdm import tqdm
import soundfile as sf
import DynamicInput as dp
from tensorflow import keras



def calc_istft(mag, phase):
        m, n = mag.shape
        complexx = []
        for i in range(m):
            for j in range(n):
                complexx.append(cmath.rect(mag[i][j], phase[i][j]))

        r = np.array(complexx).reshape(m, n)
        return lb.istft(r)

parser = argparse.ArgumentParser(description="CAE Test Script")

# Add dataset path
parser.add_argument("-dataset", "--path", type=str, default="./processed_data/", help="Dataset Path (Pre-processed). default path is ./processed_data/")
parser.add_argument("-v", "--vpath", type=str, help="vocals model path")
parser.add_argument("-b", "--bpath", type=str, help="bass model path")
parser.add_argument("-d", "--dpath", type=str, help="drums model path")
parser.add_argument("-o", "--opath", type=str, help="other model path")
args = parser.parse_args()


dest_path = "./results/"

print("Loading Models")
try:
    vocal = keras.models.load_model(args.vpath)
    drums = keras.models.load_model(args.bpath)
    bass = keras.models.load_model(args.dpath)
    other = keras.models.load_model(args.opath)
except:
    print("Issues with the given model paths. Kindly check.")
    exit(1)

bfiles = sorted(os.listdir(args.path))

for i in tqdm(bfiles):
    v, fs = lb.load(args.path+i+'/vocals.wav')
    b, fs = lb.load(args.path+i+'/bass.wav')
    d, fs = lb.load(args.path+i+'/drums.wav')
    o, fs = lb.load(args.path+i+'/other.wav')
    

    stft_v = lb.stft(v)
    stft_b = lb.stft(b)
    stft_d = lb.stft(d)
    stft_o = lb.stft(o)

    inp_v = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft_v), 3, 1)
    inp_b = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft_b), 3, 1)
    inp_d = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft_d), 3, 1)
    inp_o = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft_o), 3, 1)


    predict_v = vocal.predict(inp_v, verbose=1)
    predict_b = bass.predict(inp_b, verbose=1)
    predict_d = drums.predict(inp_d, verbose=1)
    predict_o = other.predict(inp_o, verbose=1)

    predict_v = np.squeeze(predict_v, axis=3)
    predict_b = np.squeeze(predict_b, axis=3)
    predict_d = np.squeeze(predict_d, axis=3)
    predict_o = np.squeeze(predict_o, axis=3)

    decode_v = dp.DynamicInput.decode(dp.DynamicInput, predict_v, 3, 1)
    decode_b = dp.DynamicInput.decode(dp.DynamicInput, predict_b, 3, 1)
    decode_d = dp.DynamicInput.decode(dp.DynamicInput, predict_d, 3, 1)
    decode_o = dp.DynamicInput.decode(dp.DynamicInput, predict_o, 3, 1)

    reconstruct_v = calc_istft(decode_v, np.angle(stft_v))
    reconstruct_b = calc_istft(decode_b, np.angle(stft_b))
    reconstruct_d = calc_istft(decode_d, np.angle(stft_d))
    reconstruct_o = calc_istft(decode_o, np.angle(stft_o))

    out_path = dest_path + i
    os.makedirs(out_path, exist_ok=True)

    sf.write(out_path + '/vocals.wav', reconstruct_v, fs)
    sf.write(out_path + '/bass.wav', reconstruct_b, fs)
    sf.write(out_path + '/drums.wav', reconstruct_d, fs)
    sf.write(out_path + '/other.wav', reconstruct_o, fs)
