import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
from scipy import signal
from tac2persian.utils.display import stream, progbar
from tac2persian.models.modules_wavernn import *


class WaveRNN(nn.Module) :
    def __init__(self, **params):
        super().__init__()
        self.params = params
        self.mode = self.params["voc_mode"]
        self.pad = self.params["pad"]
        self.use_upsample_net = self.params["use_upsample_net"]
        self.use_aux_net = self.params["use_aux_net"]
        self.upsample_factors = self.params["upsample_factors"]
        self.feat_dims = self.params["audio"]["num_mels"]
        self.compute_dims = self.params["compute_dims"]
        self.res_out_dims = self.params["res_out_dims"]
        self.res_blocks = self.params["res_blocks"]
        self.rnn_dims = self.params["rnn_dims"]
        self.fc_dims = self.params["fc_dims"]
        self.aux_dims = self.res_out_dims // 4
        self.hop_length = self.params["audio"]["hop_length"]
        self.sample_rate = self.params["audio"]["sample_rate"]

        if self.mode == 'MOL':
            self.n_classes = 3 * 10
        else:
            raise RuntimeError("Uknown mode.")

        if self.use_upsample_net:
            assert np.cumproduct(self.upsample_factors)[-1] == self.hop_length, " [!] upsample scales needs to be equal to hop_length"
            self.upsample = UpsampleNetwork(self.feat_dims, self.upsample_factors, self.compute_dims,
                                            self.res_blocks, self.res_out_dims, self.pad, self.use_aux_net)
        else:
            self.upsample = Upsample(self.hop_length, self.pad, self.res_blocks, self.feat_dims, self.compute_dims, self.res_out_dims, self.use_aux_net)
        
        if self.use_aux_net:
            self.I = nn.Linear(self.feat_dims + self.aux_dims + 1, self.rnn_dims)
            self.rnn1 = nn.GRU(self.rnn_dims, self.rnn_dims, batch_first=True)
            self.rnn2 = nn.GRU(self.rnn_dims + self.aux_dims, self.rnn_dims, batch_first=True)
            self.fc1 = nn.Linear(self.rnn_dims + self.aux_dims, self.fc_dims)
            self.fc2 = nn.Linear(self.fc_dims + self.aux_dims, self.fc_dims)
            self.fc3 = nn.Linear(self.fc_dims, self.n_classes)
        else:
            self.I = nn.Linear(self.feat_dims + 1, self.rnn_dims)
            self.rnn1 = nn.GRU(self.rnn_dims, self.rnn_dims, batch_first=True)
            self.rnn2 = nn.GRU(self.rnn_dims, self.rnn_dims, batch_first=True)
            self.fc1 = nn.Linear(self.rnn_dims, self.fc_dims)
            self.fc2 = nn.Linear(self.fc_dims, self.fc_dims)
            self.fc3 = nn.Linear(self.fc_dims, self.n_classes)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))


    def forward(self, x, mels) :
        device = next(self.parameters()).device
        self.step += 1
        bsize = x.size(0)
        h1 = torch.zeros(1, bsize, self.rnn_dims).to(device)
        h2 = torch.zeros(1, bsize, self.rnn_dims).to(device)
        mels, aux = self.upsample(mels)

        if self.use_aux_net:
            aux_idx = [self.aux_dims * i for i in range(5)]
            a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
            a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
            a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
            a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2) if self.use_aux_net else torch.cat([x.unsqueeze(-1), mels], dim=2)
        x = self.I(x)
        res = x
        self.rnn1.flatten_parameters()
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2) if self.use_aux_net else x
        self.rnn2.flatten_parameters()
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=2) if self.use_aux_net else x
        x = F.relu(self.fc1(x))

        x = torch.cat([x, a4], dim=2) if self.use_aux_net else x
        x = F.relu(self.fc2(x))
        return self.fc3(x)


    def generate(self, mels, batched, target, overlap) :
        device = next(self.parameters()).device
        self.eval()
        output = []
        start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad() :
            wave_len = (mels.size(-1) - 1) * self.hop_length
            mels = self.pad_tensor(mels.transpose(1, 2), pad=self.pad, side='both')
            mels, aux = self.upsample(mels.transpose(1, 2))

            if batched :
                mels = self.fold_with_overlap(mels, target, overlap)
                if aux is not None:
                    aux = self.fold_with_overlap(aux, target, overlap)

            b_size, seq_len, _ = mels.size()

            h1 = torch.zeros(b_size, self.rnn_dims).to(device)
            h2 = torch.zeros(b_size, self.rnn_dims).to(device)
            x = torch.zeros(b_size, 1).to(device)

            if self.use_aux_net:
                d = self.aux_dims
                aux_split = [aux[:, :, d*i:d*(i+1)] for i in range(4)]

            for i in range(seq_len) :

                m_t = mels[:, i, :]

                if self.use_aux_net:
                    a1_t, a2_t, a3_t, a4_t = \
                        (a[:, i, :] for a in aux_split)

                x = torch.cat([x, m_t, a1_t], dim=1) if self.use_aux_net else torch.cat([x, m_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1) if self.use_aux_net else x
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1) if self.use_aux_net else x
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1) if self.use_aux_net else x
                x = F.relu(self.fc2(x))

                logits = self.fc3(x)

                if self.mode == 'MOL':
                    sample = sample_from_discretized_mix_logistic(logits.unsqueeze(0).transpose(1, 2))
                    output.append(sample.view(-1))
                    x = sample.transpose(0, 1).to(device)
                elif self.mode == 'GAUSS':
                    sample = sample_from_gaussian(logits.unsqueeze(0).transpose(1, 2))
                    output.append(sample.view(-1))
                    x = sample.transpose(0, 1).to(device)
                elif self.mode == 'CLASSIFICATION':
                    posterior = F.softmax(logits, dim=1)
                    distrib = torch.distributions.Categorical(posterior)

                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                    output.append(sample)
                    x = sample.unsqueeze(-1)
                else:
                    raise RuntimeError("Unknown model mode value - ", self.mode)

                if i % 100 == 0 : self.gen_display(i, seq_len, b_size, start)

        output = torch.stack(output).transpose(0, 1)
        output = output.cpu().numpy()
        output = output.astype(np.float64)

        if batched :
            output = self.xfade_and_unfold(output, target, overlap)
        else :
            output = output[0]

        output = output[:wave_len]

        self.train()
        return output

    def gen_display(self, i, seq_len, b_size, start) :
        gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
        realtime_ratio = gen_rate * 1000 / self.sample_rate
        message = '%i/%i -- batch_size: %i -- gen_rate: %.1f kHz -- x_realtime: %.1f  ' % (i * b_size, seq_len * b_size, b_size, gen_rate, realtime_ratio)
        stream(message)

    def get_gru_cell(self, gru) :
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell


    def pad_tensor(self, x, pad, side='both') :
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        device = next(self.parameters()).device
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        padded = torch.zeros(b, total, c).to(device)
        if side == 'before' or side == 'both' :
            padded[:, pad:pad+t, :] = x
        elif side == 'after' :
            padded[:, :t, :] = x
        return padded


    def fold_with_overlap(self, x, target, overlap):
        device = next(self.parameters()).device
        ''' Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()
        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup
        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)
        Details:
            x = [[h1, h2, ... hn]]
            Where each h is a vector of conditioning features
            Eg: target=2, overlap=1 with x.size(1)=10
            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        '''

        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')

        folded = torch.zeros(num_folds, target + 2 * overlap, features).to(device)

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded

    def xfade_and_unfold(self, y, target, overlap):

        ''' Applies a crossfade and unfolds into a 1d array.
        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup
        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64
        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]
            Apply a gain envelope at both ends of the sequences
            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]
            Stagger and add up the groups of samples:
            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]
        '''

        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the rnn warmup
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len), dtype=np.float64)

        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len), dtype=np.float64)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    def get_step(self):
        return self.step.data.item()
