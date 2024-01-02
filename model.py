import torch
import torch.nn as nn
import math

# Input embedding
# Each word corresponds to an input id (position in vocabulary)
# Each input id corresponds to an vector of size d_model

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
# Position embedding
# Each position in the sentence is mapped to a positional encoding given by formulas listed in the paper
# We then create a positional embedding by adding the input embedding and the positional encoding

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Matrix (seq_len, d_model)
        positional_encoding = torch.zeros(seq_len, d_model) 
        # Vector (seq_len, 1)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions and apply the cos to odd positions
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        # (1, seq_len, d_model)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)
    
    def forward(self, x):
        x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
# Layer normalization
# We apply layer normalization after each of the layers in the transformer e.g. feed forward layer and multi-head attention layer

class LayerNormalization(nn.Module):

    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return (self.alpha * (x - mean) / (std + self.epsilon)) + self.bias

# Feed forward network
# A network consisting of two linear transformations with a ReLU activation in between
# This is applied in both the end of the encoder and decoder

class FeedForwardNet(nn.Module):

    def __init__(self, d_model: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_hidden) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_hidden, d_model) # W2 and B2
    
    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_hidden) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# Multi-head attention
# Input (seq_len, d_model) -> Three copies K, V, Q of same size and values
# Weights W^K, W^V, W^Q: (d_model, d_model)
# Obtain K', V', Q': (seq_len, d_model) * (d_model, d_model) -> (seq_len, d_model)
# Split into different heads: Q1, Q2, ..., Qk, V1, V2, ..., Vk, K1, K2, ..., Kk all of size (seq_len, d_model / k)
# Compute attention for each head: head_i = Attention(Qi, Ki, Vi)
# Concatenate the heads: multihead = Concat(head_1, head_2, ..., head_k) => (seq_len, d_model)
# Multiply the multihead with W_0

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model) # WQ
        self.w_k = nn.Linear(d_model, d_model) # WK
        self.w_v = nn.Linear(d_model, d_model) # WV

        self.w_o = nn.Linear(d_model, d_model) #W0
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch_len, num_heads, seq_len, d_k) -> (batch, num_heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (batch, num_heads, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
        
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)

# Residual connection
# When we receive the output of a particular layer e.g. FFN and multi-head attention, we add the output to the input of the layer
# after normalizing the output

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Encoder
# Obtain the encoder as described in the diagram within the paper using the building blocks above

class EncoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, feed_forward_net: FeedForwardNet, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward_net = feed_forward_net
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, source_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, source_mask))
        x = self.residual_connections[1](x, self.feed_forward_net)
        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Decoder
# Obtain the decoder as described in the diagram within the paper using the building blocks above

class DecoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward_net: FeedForwardNet, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward_net = feed_forward_net
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, source_mask))
        x = self.residual_connections[2](x, self.feed_forward_net)
        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)

# Projection layers
# Given the output of the decoder, we now want to convert this is a probabalistic interpretation to produce a final output sentence

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim = -1)

# Transformer
# Combine all the building blocks above to construct the transformer model

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, source_embedding: InputEmbeddings, 
                 target_embedding: InputEmbeddings, source_position: PositionalEncoding, 
                 target_position: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_position = source_position
        self.target_position = target_position
        self.projection_layer = projection_layer
    
    def encode(self, source, source_mask):
        source = self.source_embedding(source)
        source = self.source_position(source)
        return self.encoder(source, source_mask)

    def decode(self, encoder_output, source_mask, target, target_mask):
        target = self.target_embedding(target)
        target = self.target_position(target)
        return self.decoder(target, encoder_output, source_mask, target_mask)
    
    def project(self, x):
        return self.projection_layer(x)

# Method to construct the transformer model

def build_transformer(source_vocab_size: int, target_vocab_size: int, source_seq_len: int, target_seq_len: int, d_model: int = 512,
                      N: int = 6, num_heads: int = 8, dropout: float = 0.1, d_hidden: int = 2048) -> Transformer:
    # Create the embedding layers
    source_embedding = InputEmbeddings(d_model, source_vocab_size)
    target_embedding = InputEmbeddings(d_model, target_vocab_size)

    # Create the positional encoding layers
    source_position = PositionalEncoding(d_model, source_seq_len, dropout)
    target_position = PositionalEncoding(d_model, target_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        feed_forward_net = FeedForwardNet(d_model, d_hidden, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_net, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        feed_forward_net = FeedForwardNet(d_model, d_hidden, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_net, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, source_embedding, target_embedding, source_position, target_position, projection_layer)

    # Initialize the parameters
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    
    return transformer