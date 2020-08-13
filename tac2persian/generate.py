import torch
import numpy as np
import librosa
import argparse
from tac2persian.utils.generic import load_config
from tac2persian.utils.g2p.g2p import Grapheme2Phoneme
from tac2persian.models.tacotron2 import Tacotron2
from tac2persian.models.wavernn import WaveRNN
from tac2persian.utils.g2p.char_list import char_list as char_list_g2p


# ========== Tacotron
def get_tacotron(config_path, checkpoint_path, device):
    # Init model
    params_tacotron= load_config(config_path)
    params_tacotron["num_chars"] = len(char_list_g2p)
    tacotron = Tacotron2(**params_tacotron).to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    tacotron.load_state_dict(state_dict)
    print("Loaded Tacotron checkpoint.")

    return tacotron, params_tacotron


def generate_melspec(tacotron, params_tacotron, g2p, input_text, lang, spk_id, device):
    inp_chars = torch.tensor(g2p.text_to_sequence(input_text, language=lang)).long().to(device)
    # Feed inputs to the models
    postnet_outputs, attn_weights = tacotron.generate(inp_chars, spk_id)
    mel = postnet_outputs.T
    
    return mel, attn_weights


# ========== WaveRNN
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


# ========== Main
def main(args):
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Generate melspec
    tacotron, params_tacotron = get_tacotron(args.tacotron_config_path, 
                                             args.tacotron_checkpoint_path, 
                                             device)
    g2p = Grapheme2Phoneme()
    spk_id = None
    melspec, attn_weights = generate_melspec(tacotron, 
                                             params_tacotron, 
                                             g2p, 
                                             args.inp_text, 
                                             args.lang, 
                                             spk_id, 
                                             device)
    
    # Generate waveform
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
    parser.add_argument("--inp_text", type=str)
    parser.add_argument("--lang", default="fa", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    main(args)
