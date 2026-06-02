import torch.nn as nn
import torch
class Condional_LayerNorm(nn.Module):

    def __init__(self,
                normal_shape,
                epsilon=1e-5
                ):
        super(Condional_LayerNorm, self).__init__()
        if isinstance(normal_shape, int):
            self.normal_shape = normal_shape
        self.speaker_embedding_dim = 256 # 768 for whisper
        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        self.W_bias = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)
    
    def forward(self, x, speaker_embedding):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= scale
        y += bias
        # y *= scale.unsqueeze(1)
        # y += bias.unsqueeze(1)

        return y