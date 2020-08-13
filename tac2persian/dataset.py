import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import random
import os
import pickle
import numpy as np
import random
from tac2persian.utils.g2p.g2p import Grapheme2Phoneme


class TTSDataset(Dataset):
    def __init__(self, config, eval=False):
        self.config = config
        self.eval = eval
        self.g2p = Grapheme2Phoneme()
        self._load_metadata()

    def _convert_to_list(self, idx_text):
        r"""Opens melspec file (.npy format) and returns its length."""
        idx_text = map(int, idx_text.split(','))
        idx_text = [a for a in idx_text]
        return idx_text
        
    def _load_ds_metadata(self, ds):
        r"""Loads dataset metafile and returns a dictionary 
        with filenames as its keys."""
        if self.eval:
            meta_file = ds["eval_metafile"]
        else:
            meta_file = ds["train_metafile"]
        
        # Load DS metadat
        metafile_path = os.path.join(ds["dataset_path"], meta_file)
        dataset_path = ds["dataset_path"]
        with open(metafile_path) as metadata:
            all_lines = metadata.readlines()
        all_lines = [l.strip() for l in all_lines]
        all_lines = [l.split("|") for l in all_lines]
        
        # Create metadata dict
        # Each key, value pair is as below:
        # filename:{ds_root, speaker, transcript, transcript_phonemized, mel_len, inp_chars_idx}
        metadata = {l[0]:{"ds_root": dataset_path, "speaker": l[1], "transcript": l[2], 
                          "transcript_phonemized":l[3], 'mel_len': int(l[4]), 
                          'inp_chars_idx': self._convert_to_list(l[5])} for l in all_lines}
                
        # Remove items whose speaker is not in the speakers' list
        metadata = {k:v for (k,v) in metadata.items() if v["speaker"] in ds["speakers_list"]}
        
        # Extend global list of speakers
        self.speakers_list.extend(ds["speakers_list"])

        # Remove long mels
        max_mel_len = ds["max_mel_len"]
        print(f"Initially found {len(metadata.keys())} items in the dataset.")
        metadata = {k:v for (k,v) in metadata.items() if v["mel_len"] < max_mel_len}
        print(f"Number of items in the dataset after removing long mels :{len(metadata.keys())}\n")
       
        return metadata

    def _load_metadata(self):
        self.metadata = {}
        self.speakers_list =[]
        for ds in self.config["datasets"].keys():
            print(f"Loading data from {ds}")
            ds_meta = self._load_ds_metadata(self.config["datasets"][ds])
            self.metadata.update(ds_meta)

        # Speaker ID
        self.speaker_to_id = {s:i for (i,s) in enumerate(self.speakers_list)}
        self.id_to_speaker = {b:a for (a,b) in self.speaker_to_id.items()}
        
        # Items' list
        self.items = list(self.metadata.keys())

    def get_mel_lengths(self):
        mel_lengths = [self.metadata[k]["mel_len"] for k in self.items]
        return mel_lengths

    def get_sample_by_idx(self, index):
        item_id = self.items[index]
        mel = np.load(os.path.join(self.metadata[item_id]["ds_root"], 'melspecs', f'{item_id}.npy'))
        
        return mel, item_id, self.metadata[item_id]

    def __getitem__(self, index):
        item_id = self.items[index]
        
        # Get input chars (sequence of indices)
        inp_chars = self.metadata[item_id]["inp_chars_idx"]
        
        # Get speaker ID
        speaker_id = self.speaker_to_id[self.metadata[item_id]["speaker"]]
            
        # Load mel
        mel = np.load(os.path.join(self.metadata[item_id]["ds_root"], 'melspecs', f'{item_id}.npy'))
        mel_len = self.metadata[item_id]["mel_len"]
        
        return item_id, inp_chars, mel, mel_len, speaker_id

    def __len__(self):
        return len(self.items)

        
# ==================================== Data loader
def get_tacotron2_dataloader(config,
                             eval=False, 
                             binned_sampler=True):
    dataset = TTSDataset(config, eval=eval)
    mel_lengths = dataset.get_mel_lengths()
    sampler = BinnedLengthSampler(mel_lengths, config["batch_size"], config["batch_size"] * 3)
    dataloader = DataLoader(dataset,
                            collate_fn=lambda batch: _collate_tts(batch, config["model"]["max_reduction_factor"]),
                            batch_size=config["batch_size"],
                            sampler=sampler,
                            num_workers=config["num_workers"],
                            drop_last=False,
                            pin_memory=True,
                            shuffle=False)
    idx_longest = mel_lengths.index(max(mel_lengths))
    
    # Used to evaluate attention during training process
    attn_example = dataset.get_sample_by_idx(idx_longest)

    return dataloader, attn_example, dataset.speaker_to_id


# ==================================== Collate function      
def _collate_tts(batch, r):
    #           0        1        2      3         4      
    # batch: item_id, inp_chars, mel, mel_len, speaker_id

    # Compute text lengths and 
    text_lenghts = np.array([len(d[1]) for d in batch])
    
    # Sort items with text input length for RNN efficiency
    text_lenghts, ids_sorted_decreasing = torch.sort(torch.LongTensor(text_lenghts), dim=0, descending=True)

    # Create list of batch items sorted by text length
    item_ids = [batch[idx][0] for idx in ids_sorted_decreasing]
    text = [np.array(batch[idx][1]) for idx in ids_sorted_decreasing]
    mel = [np.array(batch[idx][2]) for idx in ids_sorted_decreasing]
    mel_lengths = [batch[idx][3] for idx in ids_sorted_decreasing]
    speaker_ids = [batch[idx][4] for idx in ids_sorted_decreasing]

    # Compute 'stop token' targets
    stop_targets = [np.array([0.] * (mel_len - 1) + [1.]) for mel_len in mel_lengths]

    # PAD stop targets
    stop_targets = prepare_stop_target(stop_targets, r)

    # PAD sequences with longest instance in the batch
    text = prepare_text(text).astype(np.int32)

    # PAD mel-specs
    mel = prepare_spec(mel, r)

    # B x D x T --> B x T x D
    mel = mel.transpose(0, 2, 1)

    # Convert numpy arrays to PyTorch tensor
    text = torch.LongTensor(text)
    text_lenghts = torch.LongTensor(text_lenghts)
    mel = torch.FloatTensor(mel).contiguous()
    mel_lengths = torch.LongTensor(mel_lengths)
    speaker_ids = torch.tensor(speaker_ids).long()
    stop_targets = torch.FloatTensor(stop_targets)

    return item_ids, text, text_lenghts, mel, mel_lengths, speaker_ids, stop_targets


# Pad text
def _pad_text(x, 
              length):
    _pad = 0
    assert x.ndim == 1
    return np.pad(
        x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def prepare_text(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_text(x, max_len) for x in inputs])


# Pad spectrogram
def _pad_spec(x, 
              length):
    _pad = 0
    assert x.ndim == 2
    x = np.pad(
        x, [[0, 0], [0, length - x.shape[1]]],
        mode='constant',
        constant_values=_pad)
    return x


def prepare_spec(inputs, 
                 out_steps):
    max_len = max((x.shape[1] for x in inputs))
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_spec(x, pad_len) for x in inputs])


# Pad stop target
def _pad_stop_target(x, length):
    _pad = 1.
    assert x.ndim == 1
    return np.pad(
        x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def prepare_stop_target(inputs, 
                        out_steps):
    """ Pad row vectors with 1. """
    max_len = max((x.shape[0] for x in inputs))
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_stop_target(x, pad_len) for x in inputs])


# ==================================== Binned sampler   
class BinnedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths).long())
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)

