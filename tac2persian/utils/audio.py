import os
import numpy as np
from scipy import signal, io
import librosa
import soundfile as sf


def _stft(x, 
         n_fft, 
         n_shift, 
         win_length=None, 
         window="hann", 
         center=True, 
         pad_mode="reflect"):
    r"""Computes Short-time Fourier transform for a given audio signal."""
    # x: [Time, Channel]
    if x.ndim == 1:
        single_channel = True
        # x: [Time] -> [Time, Channel]
        x = x[:, None]
    else:
        single_channel = False
    x = x.astype(np.float32)

    # x: [Time, Channel, Freq]
    x = np.stack([librosa.stft(x[:, ch],
                               n_fft=n_fft,
                               hop_length=n_shift,
                               win_length=win_length,
                               window=window,
                               center=center,
                               pad_mode=pad_mode,).T for ch in range(x.shape[1])], axis=1,)
    if single_channel:
        # x: [Time, Channel, Freq] -> [Time, Freq]
        x = x[:, 0]

    return x


def _stft2logmelspectrogram(x_stft, 
                           fs, 
                           n_mels, 
                           n_fft, 
                           fmin=None, 
                           fmax=None, 
                           eps=1e-10):
    r"""Converts STFT to log-melspectrogram."""
    # x_stft: (Time, Channel, Freq) or (Time, Freq)
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax

    # spc: (Time, Channel, Freq) or (Time, Freq)
    spc = np.abs(x_stft)
    # mel_basis: (Mel_freq, Freq)
    mel_basis = librosa.filters.mel(fs, n_fft, n_mels, fmin, fmax)
    # lmspc: (Time, Channel, Mel_freq) or (Time, Mel_freq)
    lmspc = np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

    return lmspc


def log_melspectrogram(x, 
                       sample_rate,
                       n_fft,
                       hop_length,
                       win_length,
                       num_mels,
                       mel_fmin,
                       mel_fmax,
                       window="hann",
                       pad_mode="reflect"):
    r"""Computes log melspectrogram of an audio signal."""
    # Compute STFT
    x_stft = _stft(x,
                   n_fft=n_fft,
                   n_shift=hop_length,
                   win_length=win_length,
                   window="hann",
                   pad_mode="reflect",)

    # Compute log-melspec
    return _stft2logmelspectrogram(x_stft, 
                                   fs=sample_rate, 
                                   n_mels=num_mels, 
                                   n_fft=n_fft, 
                                   fmin=mel_fmin, 
                                   fmax=mel_fmax, 
                                   eps=1e-10).T


def trim_silence(wav, ref_level_db):
    r"""Trims margin silent."""
    return librosa.effects.trim(wav, 
                                top_db=ref_level_db, 
                                frame_length=1024, 
                                hop_length=256)[0]