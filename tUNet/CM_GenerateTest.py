import numpy as np
import os
from tqdm import tqdm
import pyroomacoustics as pra


def RoomImpulseResponse(source_signals, room_dim, delay_time):
    
    vocals, bass, drums, other = source_signals[0], source_signals[1], source_signals[2], source_signals[3]
    fs = 22050
    rt60_tgt = 0.25 #seconds reverberation
    
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # Microphone positions [x, y, z] in meters
    source_pos = np.array([[3, 5, 1], [5.5, 5.5, 1], [8, 5.5, 1], [10.5, 5, 1]])

    # Sound source positions [x, y, z] in meters
    mic_pos = np.array([[3, 4.5, 1], [5.5, 5, 1], [8, 5, 1], [10.5, 4.5, 1]]).T

    # Create a ShoeBox room
    #m = pra.Material(energy_absorption="hard_surface")
    #pra.Material(e_absorption)
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
    #room = pra.ShoeBox(room_dim, fs=fs)
    
    room.add_source(source_pos[0], signal=drums, delay=delay_time)
    room.add_source(source_pos[1], signal=vocals, delay=delay_time)
    room.add_source(source_pos[2], signal=bass, delay=delay_time)
    room.add_source(source_pos[3], signal=other, delay=delay_time)
    
    # Add microphone array to the room
    mic_array = pra.MicrophoneArray(mic_pos, fs=fs)
    room.add_microphone_array(mic_array)
    
    room.simulate()
    room.compute_rir()

    result = []
    for i, mic in enumerate(mic_array.R.T):
        mic_signal = mic_array.signals[i]
        result.append(mic_signal)
    
    return result


dpath = "/home/anchal/Desktop/rajesh/Clean/Dataset/MUSDBHQ/LMTest/"
outpath = "/home/anchal/Desktop/rajesh/Clean/dfUNet/FourSource/"


print('(1/4) Loading Files...')
#x1 = np.load(dpath+'xtest.npy')
y1 = np.load(dpath+'ytest.npy')
#x2 = np.load(dpath+'xtest_lh.npy')
y2 = np.load(dpath+'ytest_lh.npy')

#Xtest = np.vstack([x1, x2])
Ytest = np.vstack([y1, y2])
print(Ytest.shape)

#Xtest = Xtest[:, :, :220448]
Ytest = Ytest[:, :, :220448]
print(Ytest.shape)




rir_songs = []
for si in tqdm(Ytest[:, :, :]):
    # Room dimensions [length, width, height] in meters
    #room_dim = [np.random.randint(10, 15), np.random.randint(10, 15), np.random.randint(8, 12)]
    room_dim = [13.5, 7, 7]
    #delay_time = np.random.randint(1, 5)*1e-3 #20-50ms
    delay_time = 1e-6
    same_song_outs = RoomImpulseResponse(si, room_dim, delay_time)
    v, b, d, o = same_song_outs[1], same_song_outs[2], same_song_outs[0], same_song_outs[3]
    rir_songs.append([v, b, d, o])


np.save('CM_Xtest.npy', rir_songs)

