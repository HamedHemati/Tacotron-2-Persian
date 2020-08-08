import torch
import torch.nn as nn
import numpy as np
from tac2persian.models.modules_tacotron2 import Encoder, Decoder
 

class Tacotron2(nn.Module):
    def __init__(self, **params):
        super(Tacotron2, self).__init__()
        self.params = params
        self._init_model()
        
    def _init_model(self):
        encoder_out_dim = self.params["enc_blstm_hidden_size"]
        
        # ----------------- Speaker embedding
        if self.params["use_spk_emb"]:
            self.spk_emb = nn.Embedding(self.config["num_spk"], self.config["spk_emb_size"])
            encoder_out_dim += self.params["spk_emb_size"]

        # ----------------- Encoder and decoder
        self.encoder = Encoder(self.params)
        self.decoder = Decoder(self.params, encoder_out_dim)
    
        # ----------------- Step buffer
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
    
    def get_num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        return num_parameters

    def forward(self, 
                inp_chars, 
                len_chars, 
                mels, 
                mel_len, 
                spk_ids):
        if self.training: 
            self.step += 1    

        # Feed input chars to the encoder
        encoder_outputs = self.encoder(inp_chars, len_chars)
        
        # Concatenate speaker embedding
        if self.params["use_spk_emb"]:
            spk_emb_vec = self.spk_emb(spk_ids).unsqueeze(1).expand(encoder_outputs.size(0), encoder_outputs.size(1), -1)
            encoder_outputs = torch.cat([encoder_outputs, spk_emb_vec], dim=-1)

        # Feed the encoder outputs to the decoder
        postnet_outputs, outputs, stop_values, attn_weights = self.decoder(encoder_outputs, mels)
        
        return postnet_outputs, outputs, stop_values, attn_weights

    def generate(self, 
                 inp_chars, 
                 spk_id, 
                 additional_inputs=None):
        self.eval()

        # Feed input chars to the encoder
        encoder_outputs = self.encoder.generate(inp_chars)
        
        # Apply speaker embedding
        if self.params["use_spk_emb"]:
            spk_emb_vec = self.spk_emb(spk_id).unsqueeze(1).expand(encoder_outputs.size(0), encoder_outputs.size(1), -1)
            encoder_outputs = torch.cat([encoder_outputs, spk_emb_vec], dim=-1)

        # Feed the encoder outputs to the decoder
        postnet_outputs, _, attn_weights = self.decoder.generate(encoder_outputs)
        
        attn_weights = torch.cat(attn_weights, 1).squeeze(0)
        
        self.train()
        
        return postnet_outputs.cpu().data.numpy(), attn_weights.cpu().data.numpy()
    
    def set_reduction_factor(self, r):
        self.decoder.set_reduction_factor(r)
    
    def get_reduction_factor(self):
        return self.decoder.get_reduction_factor()

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        self.step = self.step.data.new_tensor(1)