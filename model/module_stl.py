import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F

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

class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out