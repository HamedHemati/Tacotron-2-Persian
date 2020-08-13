import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_attention(attn, path):
    """Plot attention."""
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(attn.T, interpolation='nearest', aspect='auto')
    fig.savefig(str(f'{path}.png'), bbox_inches='tight')
    plt.close(fig)


def plot_spectrogram(M, path, length=None):
    """Plot spectrogram."""
    M = np.flip(M, axis=0)
    if length: 
        M = M[:, :length]
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(M, interpolation='nearest', aspect='auto')
    fig.savefig(str(f'{path}.png'), bbox_inches='tight')
    plt.close(fig)