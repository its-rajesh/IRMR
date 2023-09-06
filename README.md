# Interference Reduction in Multi-track Recordings

This repository contains the Python implementation of our paper titled "Neural Networks for Interference Reduction in Multi-track Recordings," which has been accepted for publication at the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics 2023 (WASPAA 2023).

### Introduction

When recording instrument sounds in live concerts, dedicated microphones are strategically positioned to capture the intended sources. However, these microphones inevitably pick up unintended sources as well due to the absence of acoustic shielding. This results in various forms of interference, commonly referred to as leakage, bleeding, or crosstalk.

In our paper, we propose two neural network-based solutions for interference reduction:
1. **Convolutional Autoencoders (CAEs)**: These models treat interference as noise and aim to eliminate it from the recordings.
2. **t-UNet (truncated UNet)**: We approach the problem as a specialized source separation task, leveraging the unique characteristics of the audio data.

For more in-depth information, please refer to our [paper]()


### Getting Started

Clone the repo and install dependencies in the virtual environment:
```
pip install -r requirements.txt
```
or directly create a conda environment using:
```
conda create --name irmr --file requirements.txt
```

## Dataset Creation
1. Download the MUSDB18HQ dataset [here](https://zenodo.org/record/3338373). Make sure the folder tree looks like this:
```
musdb18hq
├── train
│   └── A Classic Education-NightOwl
│       └── vocals.wav
│       └── bass.wav
│       └── drums.wav
│       └── other.wav
│       └── mixture.wav
│   └── subfolder...
├── test
│   └── Al James-Schoolboy Facination
│       └── vocals.wav
│       └── bass.wav
│       └── drums.wav
│       └── other.wav
│       └── mixture.wav
│   └── subfolder...

```
2. Creating artificial interference among the stems in each track by linear mixtures.


Navigate to the CAE or tUNet folder,
```
python ArtificialMix.py --dataset /path/to/musdb18hq/dataset/
```
The code returns two numpy files, ```Xtrain.npy``` and ```Ytrain.npy``` which will be saved in a folder ```numpy_files```.


3. Creating artificially realistic interference by introducing time delays and room impulse response


Navigate to the CAE or tUNet folder,
```
python realisticmix.py --dataset /path/to/numy_files/
```
The code returns two numpy files, ```Xtrain.npy``` and ```Ytrain.npy``` which will be saved in a folder ```realistic_mix```.

### CAEs
Training:
[More]()

### t-UNet
Training:
[More]()

will be updated soon..

