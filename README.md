# Interference Reduction in Multi-track Recordings

This repository contains the Python implementation of our paper "Neural Networks for Interference Reduction in Multi-track Recordings" accepted for publication in the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics 2023 (WASPAA 2023)

### Introduction
While recording instrument sounds in live concerts, dedicated microphones are placed to capture their corresponding source. Practically, these microphones pick up the other sources as well, as they are not acoustically shielded, leading to interference. These are also called leakage, bleeding, or crosstalk. In this paper, we have proposed two neural networks for interference reduction.

1. Convolutional Autoencoders (CAEs): Treating interference as noise
2. t-UNet (truncated UNet): Assuming problem as a special source separation problem

More details at [paper]()

### Getting Started
```
conda install requirements.txt or pip install requirements.txt
```
Download the MUSDB18HQ dataset [here](https://zenodo.org/record/3338373)

### CAEs


### t-UNet
