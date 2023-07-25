import numpy as np
import os
import librosa as lb
from tensorflow import keras
import DynamicInput as dp
import cmath
import argparse
from tqdm import tqdm
import pandas as pd
import museval


def calc_istft(mag, phase):
        m, n = mag.shape
        complexx = []
        for i in range(m):
            for j in range(n):
                complexx.append(cmath.rect(mag[i][j], phase[i][j]))

        r = np.array(complexx).reshape(m, n)
        return lb.istft(r)

def get_metrics(y):
    avg_y = []
    for i in range(len(y)):
        x = y[~np.isnan(y)]
        avg = sum(x)/(len(x)+0.00000001)
        avg_y.append(avg)
    return avg_y

def compute_sdr(true, reconstructed, fs):
    t = np.array([true])
    r = np.array([reconstructed])

    sdr, isr, sir, sar = museval.evaluate(t, r, win=fs, hop=fs)
        
    avg_sdr = get_metrics(sdr)
    avg_isr = get_metrics(isr) #Source to Spatial Distortion Image
    avg_sir = get_metrics(sir)
    avg_sar = get_metrics(sar)

    return sum(avg_sdr)/len(avg_sdr), sum(avg_isr)/len(avg_isr), sum(avg_sir)/len(avg_sir), sum(avg_sar)/len(avg_sar)


parser = argparse.ArgumentParser(description="CAE Train Script")

# Add dataset path
parser.add_argument("-testdata", "--path", type=str, help="Test dataset folder path (numpy files)")
parser.add_argument("-v", "--vpath", type=str, help="vocals model path")
parser.add_argument("-b", "--bpath", type=str, help="bass model path")
parser.add_argument("-d", "--dpath", type=str, help="drums model path")
parser.add_argument("-o", "--opath", type=str, help="other model path")

args = parser.parse_args()

dpath = args.path

print('(1/4) Loading Files...')
X_test = np.load(dpath+'Xtest.npy')
Y_test = np.load(dpath+'Ytest.npy')
print(X_test.shape, Y_test.shape)


print("Loading Models")
try:
    vocal = keras.models.load_model(args.vpath)
    drums = keras.models.load_model(args.bpath)
    bass = keras.models.load_model(args.dpath)
    other = keras.models.load_model(args.opath)
except:
    print("Issues with the given model paths. Kindly check.")
    exit(1)


Av_sdr, Av_isr, Av_sir, Av_sar = [], [], [], []
Ab_sdr, Ab_isr, Ab_sir, Ab_sar = [], [], [], []
Ad_sdr, Ad_isr, Ad_sir, Ad_sar = [], [], [], []
Ao_sdr, Ao_isr, Ao_sir, Ao_sar = [], [], [], []
Aobtained_sdr, Aobtained_isr, Aobtained_sir, Aobtained_sar = [], [], [], []
fs = 22050
L2 = []

for i in tqdm(range(100)):
    v = X_test[i][0]
    b = X_test[i][1]
    d = X_test[i][2]
    o = X_test[i][3]

    tv = Y_test[i][0]
    tb = Y_test[i][1]
    td = Y_test[i][2]
    to = Y_test[i][3]
    

    stft_v = lb.stft(v)
    stft_b = lb.stft(b)
    stft_d = lb.stft(d)
    stft_o = lb.stft(o)

    inp_v = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft_v), 1, 1)
    inp_b = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft_b), 1, 1)
    inp_d = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft_d), 1, 1)
    inp_o = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft_o), 1, 1)


    predict_v = vocal.predict(inp_v, verbose=1)
    predict_b = bass.predict(inp_b, verbose=1)
    predict_d = drums.predict(inp_d, verbose=1)
    predict_o = other.predict(inp_o, verbose=1)

    predict_v = np.squeeze(predict_v, axis=3)
    predict_b = np.squeeze(predict_b, axis=3)
    predict_d = np.squeeze(predict_d, axis=3)
    predict_o = np.squeeze(predict_o, axis=3)

    decode_v = dp.DynamicInput.decode(dp.DynamicInput, predict_v, 1, 1)
    decode_b = dp.DynamicInput.decode(dp.DynamicInput, predict_b, 1, 1)
    decode_d = dp.DynamicInput.decode(dp.DynamicInput, predict_d, 1, 1)
    decode_o = dp.DynamicInput.decode(dp.DynamicInput, predict_o, 1, 1)

    reconstruct_v = calc_istft(decode_v, np.angle(stft_v))
    reconstruct_b = calc_istft(decode_b, np.angle(stft_b))
    reconstruct_d = calc_istft(decode_d, np.angle(stft_d))
    reconstruct_o = calc_istft(decode_o, np.angle(stft_o))

    print(tv.shape, reconstruct_v.shape)

    if tv.shape[0] > reconstruct_v[0]:
        tv = tv[:reconstruct_v.shape[0]]
    else:
        reconstruct_v = reconstruct_v[:tv.shape[0]]
    
    if tb.shape[0] > reconstruct_b[0]:
        tb = tb[:reconstruct_b.shape[0]]
    else:
        reconstruct_b = reconstruct_b[:tb.shape[0]]

    if td.shape[0] > reconstruct_d[0]:
        td = td[:reconstruct_d.shape[0]]
    else:
        reconstruct_d = reconstruct_d[:td.shape[0]]

    if to.shape[0] > reconstruct_o[0]:
        to = to[:reconstruct_o.shape[0]]
    else:
        reconstruct_o = reconstruct_o[:to.shape[0]]

    print(tv.shape, reconstruct_v.shape)

    X = np.array([v, b, d, o])
    S = np.array([tv, tb, td, to])
    S_ = np.array([reconstruct_v, reconstruct_b, 
                   reconstruct_d, reconstruct_o])
    
    xn = len(v)
    sn = len(reconstruct_v)
    if xn > sn:
        X = X[:, :sn]
        S = S[:, :sn]
        S_ = S_[:, :sn]
    else:
        X = X[:, :xn]
        S = S[:, :xn]
        S_ = S_[:, :xn]
    print(X.shape, S.shape, S_.shape)
    A_act = X @ np.linalg.pinv(S)
    A_pred = X @ np.linalg.pinv(S_)

    l2norm = np.linalg.norm(A_act-A_pred)**2
    L2.append(l2norm)

    v_sdr, v_isr, v_sir, v_sar = compute_sdr(tv, reconstruct_v, fs)
    b_sdr, b_isr, b_sir, b_sar = compute_sdr(tb, reconstruct_b, fs)
    d_sdr, d_isr, d_sir, d_sar = compute_sdr(td, reconstruct_d, fs)
    o_sdr, o_isr, o_sir, o_sar = compute_sdr(to, reconstruct_o, fs)

    obtained_sdr = (v_sdr + b_sdr + d_sdr + o_sdr)/4
    obtained_isr = (v_isr + b_isr + d_isr + o_isr)/4
    obtained_sir = (v_sir + b_sir + d_sir + o_sir)/4
    obtained_sar = (v_sar + b_sar + d_sar + o_sar)/4

    Av_sdr.append(v_sdr)
    Av_isr.append(v_isr)
    Av_sir.append(v_sir)
    Av_sar.append(v_sar)

    Ab_sdr.append(b_sdr)
    Ab_isr.append(b_isr)
    Ab_sir.append(b_sir)
    Ab_sar.append(b_sar)

    Ad_sdr.append(d_sdr)
    Ad_isr.append(d_isr)
    Ad_sir.append(d_sir)
    Ad_sar.append(d_sar)

    Ao_sdr.append(o_sdr)
    Ao_isr.append(o_isr)
    Ao_sir.append(o_sir)
    Ao_sar.append(o_sar)

    Aobtained_sdr.append(obtained_sdr)
    Aobtained_isr.append(obtained_isr)
    Aobtained_sir.append(obtained_sir)
    Aobtained_sar.append(obtained_sar)

sdr = pd.DataFrame(
    {'Pred vocal sdr': Av_sdr,
     'Pred bass sdr': Ab_sdr,
     'Pred drums sdr': Ad_sdr,
     'Pred other sdr': Ao_sdr,
     'Pred Overall sdr': Aobtained_sdr,
     'Pred vocal isr': Av_isr,
     'Pred bass isr': Ab_isr,
     'Pred drums isr': Ad_isr,
     'Pred other isr': Ao_isr,
     'Pred Overall isr': Aobtained_isr,
     'Pred vocal sir': Av_sir,
     'Pred bass sir': Ab_sir,
     'Pred drums sir': Ad_sir,
     'Pred other sir': Ao_sir,
     'Pred Overall sir': Aobtained_sir,
     'Pred vocal sar': Av_sar,
     'Pred bass sar': Ab_sar,
     'Pred drums sar': Ad_sar,
     'Pred other sar': Ao_sar,
     'Pred Overall sar': Aobtained_sar,
     'Closeness':L2})

sdr.to_csv('sdr.csv')
