import torch
import torch.nn as nn
import math

class DecoderOnlyTransformer():

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length):
        super(DecoderOnlyTransformer, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.out_mlp = nn.Linear(d_model, vocab_size)


    def forward(self, x):
        
        x = self.embed(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x)

        output = self.out_mlp(x)

        return output


class DecoderLayer(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x): 
        # why do we pass x, x, x in here?
        attn_output = self.self_attn (x, x, x)