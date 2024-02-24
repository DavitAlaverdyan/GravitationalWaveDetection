import numpy as np
from nnAudio.Spectrogram import CQT1992v2
import torch
from scipy import signal

np.float = np.float64

def id2path(idx, is_train=True):
    """Generate file path based on index and dataset type.

    Args:
    idx (str): Index or identifier for the data.
    is_train (bool, optional): Flag indicating if the data belongs to the training set. Defaults to True.

    Returns:
    str: File path corresponding to the provided index and dataset type.
    """
    # Base path for the dataset
    path = "G:/W_project/Waves_data/g2net-gravitational-wave-detection"
    
    if is_train:
        # Construct path for training data
        path += "/train/" + idx[0] + "/" + idx[1] + "/" + idx[2] + "/" + idx + ".npy"
    else:
        # Construct path for test data
        path += "/test/" + idx[0] + "/" + idx[1] + "/" + idx[2] + "/" + idx + ".npy"

    return path


def increase_dimension(idx, is_train, transform=CQT1992v2(sr=2048, hop_length=64, fmin=20, fmax=500)):
    """Increase the dimensionality of a waveform and convert it to an image representation using CWT.

    Args:
    idx (str): Index or identifier for the data.
    is_train (bool): Flag indicating if the data belongs to the training set.
    transform (nnAudio.Spectrogram.CQT1992v2, optional): Transform to convert waveform to image-like representation using CWT.
        Defaults to CQT1992v2(sr=2048, hop_length=64, fmin=20, fmax=500).

    Returns:
    np.ndarray: Image-like representation of the waveform.
    """
    # Load waveform from the provided index and dataset type
    wave = np.load(id2path(idx, is_train))
    wave = np.concatenate(wave, axis=0)
    
    # Apply bandpass filter to the waveform
    bHP, aHP = signal.butter(8, (20, 500), btype='bandpass', fs=2048)
    window = signal.tukey(4096*3, 0.2)
    wave *= window
    wave = signal.filtfilt(bHP, aHP, wave)
    wave = wave / np.max(wave)
    
    # Convert waveform to a torch tensor
    wave = torch.from_numpy(wave).float()
    
    # Transform the waveform into an image-like representation using CWT
    image = transform(wave)
    
    # Convert the image to a NumPy array and transpose dimensions
    image = np.array(image)
    image = np.transpose(image, (1, 2, 0))
    
    return image