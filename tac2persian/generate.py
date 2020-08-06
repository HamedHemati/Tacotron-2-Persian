import torch
import numpy as np
import librosa
import argparse
from tac2persian.utils.generic import load_config
from tac2persian.models.wavernn import WaveRNN


def get_wavernn(config_path, checkpoint_path, device):
    # Init model
    params_wavernn = load_config(config_path)
    wavernn = WaveRNN(**params_wavernn).to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    wavernn.load_state_dict(state_dict)
    print("Loaded WaveRNN checkpoint.")

    return wavernn, params_wavernn


def generate_wav(wavernn, params_wavernn, melspec):
    # Generate audio
    batched = True
    melspec = torch.tensor(melspec).unsqueeze(0)
    out_wav = wavernn.generate(melspec, batched, params_wavernn["target"], params_wavernn["overlap"])
    return out_wav


def main(args):
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Tacotron
    melspec = None
    
    # WaveRNN
    wavernn, params_wavernn = get_wavernn(args.wavernn_config_path, args.wavernn_checkpoint_path, device)
    out_wav = generate_wav(wavernn, params_wavernn, melspec)
    librosa.output.write_wav("outputs/sample.wav", out_wav, params_wavernn["audio"]["sample_rate"])

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavernn_config_path", type=str)
    parser.add_argument("--wavernn_checkpoint_path", type=str)
    parser.add_argument("--tacotron_config_path", type=str)
    parser.add_argument("--tacotron_checkpoint_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    main(args)
