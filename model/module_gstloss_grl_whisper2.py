# adapted from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
# MIT License
#
# Copyright (c) 2018 MagicGirl Sakura
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from model.classifier import ReversalClassifier
from whisper.model import AudioEncoder
import whisper

class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, ref_enc_filters=[32, 32, 64, 64, 128, 128], n_mel_channels=80, ref_enc_gru_size=128):

        super().__init__()
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=ref_enc_filters[i])
             for i in range(K)])

        out_channels = self.calculate_channels(n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels,
                          hidden_size=ref_enc_gru_size,
                          batch_first=True)
        self.n_mel_channels = n_mel_channels
        self.ref_enc_gru_size = ref_enc_gru_size

    def forward(self, inputs, input_lengths=None):
        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels) # inputs (16, 80, 196), out (16, 1, 196, 80)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K] out (16, 4, 128, 2)
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K] out (16, 4, 256)

        if input_lengths is not None:
            input_lengths = torch.ceil(input_lengths.float() / 2 ** len(self.convs))
            input_lengths = input_lengths.cpu().numpy().astype(int)            
            out = nn.utils.rnn.pack_padded_sequence(
                        out, input_lengths, batch_first=True, enforce_sorted=False)

        self.gru.flatten_parameters()
        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, token_embedding_size//2]
    '''
    def __init__(self, token_num=4, token_embedding_size=256, num_heads=8, ref_enc_gru_size=128):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(token_num, token_embedding_size // num_heads))
        d_q = ref_enc_gru_size
        d_k = token_embedding_size // num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, num_units=token_embedding_size,
            num_heads=num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

        self.proj = nn.Linear(token_embedding_size, token_num)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1) # 16,1,128
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, token_embedding_size // num_heads]
        style_embed = self.attention(query, keys) # keys (16,3,32) query (16,1,128)
        pred_style = self.proj(style_embed)

        return style_embed, pred_style


class PredHead(nn.Module): #Add commentMore actions
    """Custom CTC head for ASR."""

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 280,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        use_gelu: bool = True,
        feats_size: int = 256
    ):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            if i == num_layers - 2:
                layers.extend(
                    [
                        nn.Linear(hidden_size, feats_size),
                        nn.GELU() if use_gelu else nn.ReLU(),
                        nn.LayerNorm(feats_size) if use_layer_norm else nn.Identity(),
                    ]
                )
            else:
                layers.extend(
                    [
                        nn.Linear(hidden_size, hidden_size),
                        nn.GELU() if use_gelu else nn.ReLU(),
                        nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity(),
                    ]
                )
        self.proj = nn.Linear(feats_size, vocab_size)
        # layers.append(nn.Linear(hidden_size, vocab_size))
        self.layers = nn.Sequential(*layers) #Add commentMore actions

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # print("hidden_states.shape", hidden_states.shape)
        feats = self.layers(hidden_states)
        logits = self.proj(feats)
        return feats, logits

class GSTWhisper(nn.Module):
    def __init__(self, n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, acc_layers, spk_layers, n_acc, n_spk, model_name):
        super().__init__()
        # self.encoder = ReferenceEncoder()
        # self.encoder = AudioEncoder(
        #     n_mels,
        #     n_audio_ctx,
        #     n_audio_state,
        #     n_audio_head,
        #     n_audio_layer
        # )

        self.model = self.load_pretrained_whisper(model_name)

        for p in self.model.encoder.parameters():
            p.requires_grad = False  # Freeze all encoder parameters
        
        del self.model.decoder

        # self.stl = STL()
        # self.grl = ReversalClassifier(
        #                         128, # n_feats
        #                         256,
        #                         222, # n_accents
        #                         0.25) # reversal_gradient_clipping = 0.25

        self.acc_head = PredHead(vocab_size=n_acc, hidden_size=n_audio_state, num_layers=acc_layers, feats_size=256)
        self.spk_head = PredHead(vocab_size=n_spk, hidden_size=n_audio_state, num_layers=spk_layers, feats_size=64)

    def forward(self, inputs, input_lengths=None):
        with torch.no_grad():
            enc_out = self.model.encoder(inputs) # (N, 128) NO length information, 1 tensor for query
        
        feats_acc, logits_acc = self.acc_head(enc_out) # (N, n_acc)
        feats_spk, logits_spk = self.spk_head(enc_out)
        
        # style_embed, pred_style = self.stl(enc_out)

        return feats_acc, logits_acc, feats_spk, logits_spk #, style_embed, pred_style

    def load_pretrained_whisper(self, model_name):
    
        whisper_model = whisper.load_model(model_name, device="cpu")  # Load the Whisper model without GPU
        model = whisper.Whisper(whisper_model.dims)  # Use the same dimensions as the original model
        model.load_state_dict(whisper_model.state_dict(), strict=False)

        return model