import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ====================================== Encoder

class Encoder(nn.Module):
    """Encoder module of Tacotron model.
    """
    def __init__(self, params):
        super(Encoder, self).__init__()
        
        self.params = params
        # Embedding layer
        self.embedding = nn.Embedding(params["num_chars"], 
                                      params["enc_emb_dim"])
        conv_init_dim = params["enc_emb_dim"]
        
        # Convolutional layers
        self.conv_residual = self.params["enc_conv_residual"]
        self.conv_layers = nn.ModuleList()
        for layer in range(params["enc_num_conv_layers"]):
            inp_channels = conv_init_dim if layer ==0 else params["enc_conv_channels"]
            if params["enc_conv_batchnorm"]:
                self.conv_layers += [torch.nn.Sequential(
                                     torch.nn.Conv1d(inp_channels, params["enc_conv_channels"], params["enc_conv_kernel_size"],
                                                     stride=1, padding=(params["enc_conv_kernel_size"]-1)//2, bias=False),
                                     torch.nn.BatchNorm1d(params["enc_conv_channels"]),
                                     torch.nn.ReLU(),
                                     torch.nn.Dropout(params["enc_conv_dropout_rate"]),)]
            else:
                self.conv_layers += [torch.nn.Sequential(
                                     torch.nn.Conv1d(inp_channels, params["enc_conv_channels"], params["enc_conv_kernel_size"], 
                                                     stride=1, padding=(params["enc_conv_kernel_size"]-1)//2, bias=False),
                                     torch.nn.ReLU(),
                                     torch.nn.Dropout(params["enc_conv_dropout_rate"]),)]
        
        # Bi-directional LSTM
        self.blstm = nn.LSTM(params["enc_conv_channels"], params["enc_blstm_hidden_size"]//2, params["enc_blstm_num_layers"],
                             batch_first=True, bidirectional=True)
        self.apply(self._init_encoder)
        
    def _init_encoder(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain("relu"))
            
    def forward(self, x, x_lens, additional_inputs=None):
        x = self.embedding(x).transpose(1, 2) # B x T x C
        
        # Conv layers    
        for i in range(len(self.conv_layers)):
            if self.conv_residual:
                x += self.conv_layers[i](x)
            else:
                x = self.conv_layers[i](x)

        # Pack the padded input sequences 
        x = pack_padded_sequence(x.transpose(1,2), x_lens, batch_first=True)
        self.blstm.flatten_parameters()
        x, _ = self.blstm(x)
        
        # Padd the blstm outputs with 0s
        x, _ = pad_packed_sequence(x, batch_first=True)
        
        return x   
    
    def generate(self, x, additional_inputs=None):
        assert len(x.size()) == 1
        x = x.unsqueeze(0)
        x_len = [x.size(1)]

        return self.forward(x, x_len, additional_inputs=additional_inputs)



# ====================================== Decoder

class ZoneOutCell(nn.Module):
    def __init__(self, cell, zoneout_rate=0.1):
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_rate = zoneout_rate
    
    def forward(self, x, prev_h):
        h = self.cell(x, prev_h)
        h = self.zoneout_(prev_h, h, self.zoneout_rate)
        return h
    
    def zoneout_(self, prev_h, h, prob):
        # Apply zoneout recursively to all h units
        if isinstance(h, tuple):
            size_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * size_h)
            return tuple([self.zoneout_(prev_h[i], h[i], prob[i]) for i in range(size_h)])
        
        if self.training:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * prev_h + (1 - mask) * h
        else:
            return prob * prev_h + (1 - prob) * h
        
        
class Prenet(nn.Module):
    def __init__(self, inp_size, num_layers=2, hidden_size=256, dropout_rate=0.5):
        super(Prenet, self).__init__()
        
        self.prenet_layers = nn.ModuleList()
        for i in range(num_layers):
            size_in = inp_size if i == 0  else hidden_size
            self.prenet_layers += [nn.Sequential(nn.Linear(size_in, hidden_size)), 
                                   nn.ReLU(),
                                   nn.Dropout(dropout_rate)]
            
    def forward(self, x):
        for i in range(len(self.prenet_layers)):
            x = self.prenet_layers[i](x)
        return x

    
class Postnet(nn.Module):
    def __init__(self, 
                 inp_size, 
                 out_size, 
                 num_conv_layers=5, 
                 conv_channels=512,
                 conv_filter_size=5, 
                 dropout_rate=0.5, 
                 conv_batch_norm=True):
        super(Postnet, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        # Conv layers 0 - (num_conv_layers - 1)
        for i in range(num_conv_layers - 1):
            inp_channels = inp_size if i == 0  else conv_channels
            out_channels = conv_channels
            if conv_batch_norm:
                self.conv_layers += [nn.Sequential(nn.Conv1d(inp_channels, 
                                                             out_channels, 
                                                             conv_filter_size,
                                                             stride=1, 
                                                             padding=(conv_filter_size - 1) // 2, 
                                                             bias=False),
                                     nn.BatchNorm1d(out_channels),
                                     nn.Tanh(),
                                     nn.Dropout(dropout_rate))]
            else:
                self.conv_layers += [nn.Sequential(nn.Conv1d(inp_channels, 
                                                             out_channels, 
                                                             conv_filter_size,
                                                             stride=1, 
                                                             padding=(conv_filter_size - 1) // 2, 
                                                             bias=False),
                                     nn.Tanh(),
                                     nn.Dropout(dropout_rate))]
        # Last conv layer
        if conv_batch_norm:
            self.conv_layers += [nn.Sequential(nn.Conv1d(conv_channels, out_size, conv_filter_size,
                                                           stride=1, padding=(conv_filter_size - 1) // 2, bias=False),
                                                  nn.BatchNorm1d(out_size),
                                                  nn.Tanh(),
                                                  nn.Dropout(dropout_rate))]
        else:
            self.conv_layers += [nn.Sequential(nn.Conv1d(conv_channels, out_size, conv_filter_size,
                                                           stride=1, padding=(conv_filter_size - 1) // 2, bias=False),
                                                  nn.Tanh(),
                                                  nn.Dropout(dropout_rate))]
    
    def forward(self, x):
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
        return x


class GMMAttentionV2(nn.Module):
    """ Discretized Graves attention: (Code from Mozilla TTS https://github.com/mozilla/TTS)
        - https://arxiv.org/abs/1910.10288
        - https://arxiv.org/pdf/1906.01083.pdf
    """
    COEF = 0.3989422917366028  # numpy.sqrt(1/(2*numpy.pi))

    def __init__(self, query_dim, K):
        super().__init__()
        self._mask_value = 1e-8
        self.K = K
        # self.attention_alignment = 0.05
        self.eps = 1e-5
        self.J = None
        self.N_a = nn.Sequential(
            nn.Linear(query_dim, query_dim//10, bias=True),
            nn.Tanh(),  # replaced ReLU with tanh
            nn.Linear(query_dim//10, 3*K, bias=True))

        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        self.init_layers()

    def init_layers(self):
        torch.nn.init.constant_(self.N_a[2].bias[(2*self.K):(3*self.K)], 1.)  # bias mean
        torch.nn.init.constant_(self.N_a[2].bias[self.K:(2*self.K)], 10)  # bias std

    def init_states(self, inputs):
        offset = 50
        if self.J is None or inputs.shape[1] + offset > self.J.shape[-1] or self.J.shape[0] != inputs.shape[0]:
            self.J = torch.arange(0, inputs.shape[1] + offset).to(inputs.device).expand_as(torch.Tensor(inputs.shape[0], self.K, inputs.shape[1] + offset))
        self.attention_weights = torch.zeros(inputs.shape[0], inputs.shape[1]).to(inputs.device)
        self.mu_prev = torch.zeros(inputs.shape[0], self.K).to(inputs.device)

    # pylint: disable=R0201
    # pylint: disable=unused-argument
    def preprocess_inputs(self, inputs):
        return None

    def forward(self, query, inputs):
        """
        Shapes:
            query: B x D_attention_rnn
            inputs: B x T_in x D_encoder
            processed_inputs: place_holder
            mask: B x T_in
        """
        gbk_t = self.N_a(query)
        gbk_t = gbk_t.view(gbk_t.size(0), -1, self.K)

        # attention model parameters
        # each B x K
        g_t = gbk_t[:, 0, :]
        b_t = gbk_t[:, 1, :]
        k_t = gbk_t[:, 2, :]
        
        # attention GMM parameters
        g_t = self.softmax(g_t) + self.eps
        mu_t = self.mu_prev + self.softplus(k_t)
        sig_t = self.softplus(b_t) + self.eps

        # update perv mu_t
        self.mu_prev = mu_t

        # calculate normalizer
        z_t = torch.sqrt(2 * np.pi * (sig_t ** 2)) + self.eps
        
        # location indices
        j = self.J[:g_t.size(0), :, :inputs.size(1)]
        
        # attention weights
        g_t = g_t.unsqueeze(2).expand(g_t.size(0), g_t.size(1), inputs.size(1))
        z_t = z_t.unsqueeze(2).expand_as(g_t)
        mu_t_ = mu_t.unsqueeze(2).expand_as(g_t)
        sig_t = sig_t.unsqueeze(2).expand_as(g_t)

        phi_t = (g_t / z_t) * torch.exp(-0.5 * (1.0 / (sig_t**2)) * (mu_t_ - j)**2)
        
        # discritize attention weights
        alpha_t = torch.sum(phi_t, 1).float()
        
        return alpha_t        


class Decoder(nn.Module):
    def __init__(self, params, encoder_out_dim):
        super(Decoder, self).__init__()
        self.params = params
        self.reduction_factor = None # to be set manually
        self.max_reduction_factor = params["max_reduction_factor"]
        
        # Set attention module
        self.attn_type = params["attn_type"]
        if self.attn_type == "gmmv2":
            self.attn = GMMAttentionV2(params["dec_lstm_hidden_size"], 
                                       params["attn_gmm_k"])
        else:
            raise RuntimeError("Attention type not defined.")

        self.output_size = params["dec_out_dim"]
        self.output_size_enc = encoder_out_dim
              
        # Prenet layers
        self.prenet = Prenet(inp_size=self.output_size, 
                             num_layers=params["dec_num_prenet_layers"],
                             hidden_size=params["dec_prenet_hidden_size"], 
                             dropout_rate=params["dec_dropout_rate"])
        
        # LSTM layers
        # ----- Attention RNN
        inp_size_rnn_atnn = self.output_size_enc + params["dec_prenet_hidden_size"]   
        self.rnn_attn = nn.LSTMCell(inp_size_rnn_atnn, params["dec_lstm_hidden_size"])
        if params["dec_zoneout_rate"] > 0.0:
            self.rnn_attn = ZoneOutCell(self.rnn_attn)      

        # ----- Decoder RNN
        self.rnn_dec = nn.LSTMCell(params["dec_lstm_hidden_size"], params["dec_lstm_hidden_size"])
        if params["dec_zoneout_rate"] > 0.0:
            self.rnn_dec = ZoneOutCell(self.rnn_dec)  
        
        # Linear projections
        inp_size_linear_proj = params["dec_lstm_hidden_size"] + self.output_size_enc # concatenates the context vector with rnn_dec's output
        self.linear_proj = nn.Linear(inp_size_linear_proj, self.output_size * self.max_reduction_factor, bias=False)
        self.stopnet = nn.Linear(inp_size_linear_proj, self.max_reduction_factor)
        
        # Postnet layers
        self.postnet = Postnet(self.output_size, self.output_size, num_conv_layers=params["dec_num_postnet_layers"],
                               conv_channels=params["dec_postnet_conv_channels"], conv_filter_size=params["dec_postnet_conv_kernel_size"],
                               dropout_rate=params["dec_dropout_rate"], conv_batch_norm=params["dec_batch_norm"])
        
        # Initialize decoder weights
        self.apply(self._init_decoder)
        
    def _init_decoder(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain("tanh"))
    
    def set_reduction_factor(self, r):
        self.reduction_factor = r
    
    def get_reduction_factor(self):
        return self.reduction_factor

    def _init_attn(self, encoder_outputs):
        if self.attn_type == "gmmv2":
            self.attn.init_states(encoder_outputs)

    def forward(self, 
                encoder_outputs, 
                mels,
                additional_inputs=None):
        device = next(self.parameters()).device
        if self.reduction_factor > 1:
            mels =  mels[:, self.reduction_factor-1::self.reduction_factor]
        
        batch_size = encoder_outputs.size(0)

        # Initialize hidden and cell states for the RNNs
        rnn_attn_h = torch.zeros(batch_size, self.rnn_attn.hidden_size).to(device)
        rnn_attn_c = torch.zeros(batch_size, self.rnn_attn.hidden_size).to(device)
        rnn_dec_h = torch.zeros(batch_size, self.rnn_dec.hidden_size).to(device)
        rnn_dec_c = torch.zeros(batch_size, self.rnn_dec.hidden_size).to(device)
        
        # Initialize the <GO> frame for prenet
        prev_out = torch.zeros(batch_size, self.output_size).to(device)
        
        # Initialize lists for outputs, stop_values and attn_weights
        outputs, stop_values, attn_weights = [], [], []
        
        # Initialize the attention network
        self._init_attn(encoder_outputs)
    
        # Iterate through all mel frames:
        for t, mel_t in enumerate(mels.transpose(0, 1)):
            # Compute prenet's output
            prenet_out = self.prenet(prev_out)
            
            # Compute attention weights for step t
            if self.attn_type == "gmmv2":
                weights_t = self.attn(rnn_attn_h, encoder_outputs)

            # Compute the context vector
            weights_t = weights_t.unsqueeze(1)
            attn_weights.append(weights_t)
            
            context_vec_t = torch.bmm(weights_t, encoder_outputs).squeeze(1)
            
            # Concatenate context vector with prenet output as input for rnn_attn
            x_attn = torch.cat([context_vec_t, prenet_out],dim=1)
            rnn_attn_h, rnn_attn_c = self.rnn_attn(x_attn, (rnn_attn_h, rnn_attn_c))
            
            # Input hidden state of rnn_attn to rnn_dec
            rnn_dec_h, rnn_dec_c = self.rnn_dec(rnn_attn_h, (rnn_dec_h, rnn_dec_c))
            
            # Concatenate rnn_dec's hidden state with the context vector as input for linear_proj and stopnet
            x_lin_proj = torch.cat([rnn_dec_h, context_vec_t], dim=1)
            outputs += [self.linear_proj(x_lin_proj).view(batch_size, self.output_size, -1)[:, :, :self.reduction_factor]]
            stop_values += [self.stopnet(x_lin_proj)[:, :self.reduction_factor]]
            
            # Update prev_out
            prev_out = mel_t
        
        # Concat tensors in the temporal dimension
        outputs = torch.cat(outputs, dim=2)
        stop_values = torch.cat(stop_values, dim=1)
        attn_weights = torch.cat(attn_weights, dim=1)
        
        # Compute output of postnet
        postnet_outputs = outputs + self.postnet(outputs)
        
        # Tranpose dimensions 1 and 2 (temporal and mel dimensions)
        outputs = outputs.transpose(1, 2)
        postnet_outputs = postnet_outputs.transpose(1, 2)
        
        return postnet_outputs, outputs, stop_values, attn_weights
    
    def generate(self, 
                 encoder_outputs, 
                 stop_threshold=0.5, 
                 minlenratio=0.0, 
                 maxlenratio=10, 
                 additional_inputs=None):
        """
            encoder_outputs: 1 x L x ENC_DIM
        """
        device = next(self.parameters()).device
        
        # Compute min and max len final melspec
        max_len = int(encoder_outputs.size(1) * maxlenratio)
        min_len = int(encoder_outputs.size(1) * minlenratio)
        
        # Initialize hidden and cell states for the RNNs
        rnn_attn_h = torch.zeros(1, self.rnn_attn.hidden_size).to(device)
        rnn_attn_c = torch.zeros(1, self.rnn_attn.hidden_size).to(device)
        rnn_dec_h = torch.zeros(1, self.rnn_dec.hidden_size).to(device)
        rnn_dec_c = torch.zeros(1, self.rnn_dec.hidden_size).to(device)

        # Initialize Go frame
        prev_out = torch.zeros(1, self.output_size).to(device)
        
        # Initialize lists for outputs, stop_values and attn_weights
        outputs, stop_values, attn_weights = [], [], []
        idx = 0
        
        # Initialize the attention network
        self._init_attn(encoder_outputs)

        while True:
            # Compute prenet's output
            prenet_out = self.prenet(prev_out)

            # Compute attention weights for step t
            if self.attn_type == "gmmv2":
                weights_t = self.attn(rnn_attn_h, encoder_outputs)

            # Compute the context vector
            weights_t = weights_t.unsqueeze(1)
            attn_weights.append(weights_t)
            context_vec_t = torch.bmm(weights_t, encoder_outputs).squeeze(1)
            
            # Concatenate context vector with prenet output as input for rnn_attn
            x_attn = torch.cat([context_vec_t, prenet_out],dim=1)
            rnn_attn_h, rnn_attn_c = self.rnn_attn(x_attn, (rnn_attn_h, rnn_attn_c))
            
            # Input hidden state of rnn_attn to rnn_dec
            rnn_dec_h, rnn_dec_c = self.rnn_dec(rnn_attn_h, (rnn_dec_h, rnn_dec_c))
            
            # Concatenate rnn_dec's hidden state with the context vector as input for linear_proj and stopnet
            x_lin_proj = torch.cat([rnn_dec_h, context_vec_t], dim=1)
            outputs += [self.linear_proj(x_lin_proj).view(1, self.output_size, -1)[:, :, :self.reduction_factor]]
            stop_values += [self.stopnet(x_lin_proj)[:, :self.reduction_factor]]
            
            # Update prev_out
            prev_out = outputs[-1][:, :, -1]
            
            # Incement idx
            idx += 1
            
            # Check whether to stop generation
            if int(sum(stop_values[-1][0])) > 0 or idx >= max_len:
                if idx < min_len:
                    continue
                outputs = torch.cat(outputs, dim=2)
                postnet_outputs = outputs + self.postnet(outputs)
                stop_values = torch.cat(stop_values, dim=0)
                
                break
        
        # Tranpose temporal and mel dimensions and squeeze dimension 0 (since batch size is 1)
        postnet_outputs = postnet_outputs.transpose(1, 2).squeeze(0)

        return postnet_outputs, stop_values, attn_weights


# ====================================== Functions
def pad_mask(mel_lens, r):
    max_len = max(mel_lens)
    remainder = max_len % r
    pad_len = max_len + (r - remainder) if remainder > 0 else max_len
    mask = [np.ones(( mel_lens[i]), dtype=bool) for i in range(len(mel_lens))]
    mask = np.stack([_pad_array(x, pad_len) for x in mask])
    return torch.tensor(mask)


def _pad_array(x, length):
    _pad = 0
    x = np.pad(
        x, [[0, length - x.shape[0]]],
        mode='constant',
        constant_values=False)
    return x