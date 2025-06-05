#
# Created By @TailsDev Or t.me/@Shoukaku07
# Initial Name: M
#
# Inspired GPT (OpenAI)
#
"""PyTorch Transformer LLM model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_size = 1024
intermediate_size = 2048
num_heads = 16
num_hidden_layer = 27
max_seq_len = 2048 # max positional embedding
vocab_size = 50257
attn_bias=False # Penambahan parameter bias
mlp_bias=False # Penambahan parameter bias or literally FFN

class RotaryEmbedding(nn.Module):
    """
    Adds rotary positional embeddings (RoPE) to a tensor.
    
    Rotary embeddings are a way to help Transformers understand the order 
    of words (or tokens) without just adding position numbers. Instead, 
    they rotate the attention inputs in a smart way.

    Args:
        dim (int): The number of dimensions to apply rotary embeddings to.
                   Usually the same as the hidden size of one attention head.
        max_seq_len (int): The max length of the input sequences this can handle.
    """
    def __init__(self, dim, max_seq_len=max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # This creates frequencies that we use to "rotate" the input.
        # The rotation helps the model know where each token is in the sequence.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        t = torch.arange(max_seq_len).float()

        freqs = torch.einsum('i , j -> i j', t, inv_freq)  # shape: [max_seq_len, dim // 2]

        self.register_buffer("cos_cached", freqs.cos()[None, :, :])  # [1, max_seq_len, dim//2]
        self.register_buffer("sin_cached", freqs.sin()[None, :, :])  # [1, max_seq_len, dim//2]

    def forward(self, x, seq_dim=1):
        """
        Applies rotary embedding to the input tensor.

        Args:
            x (Tensor): The input tensor. Shape: [..., seq_len, dim]
            seq_dim (int): Which dimension represents the sequence. Usually 1.

        Returns:
            Tensor: Same shape as input, but with rotary embedding applied.
        """
        seq_len = x.size(seq_dim)

        x1 = x[..., :seq_len, :self.dim]

        x1_reshaped = x1.view(*x1.shape[:-1], self.dim // 2, 2)

        cos = self.cos_cached[:, :seq_len, :].to(x.device)  # [1, seq_len, dim//2]
        sin = self.sin_cached[:, :seq_len, :].to(x.device)  # [1, seq_len, dim//2]

        x_even = x1_reshaped[..., 0]
        x_odd = x1_reshaped[..., 1]

        # Rotate the pairs using cos and sin like rotating a 2D point
        # This is the magic of rotary embeddings!
        x_rotated = torch.zeros_like(x1_reshaped)
        x_rotated[..., 0] = x_even * cos - x_odd * sin
        x_rotated[..., 1] = x_even * sin + x_odd * cos

        x_rotated = x_rotated.view_as(x1)

        x_out = x.clone()
        x_out[..., :seq_len, :self.dim] = x_rotated

        return x_out

class FFNLayer(nn.Module):
    """
    Feedforward Neural Network Layer used in Transformer after self-attention.

    Structure:
        Input (hidden_size)
        -> Linear(hidden_size -> intermediate_size)
        -> GELU activation
        -> Linear(intermediate_size -> hidden_size)
        -> Output (hidden_size)

    Args:
        hidden_size (int): Dimensionality of input and output features.
        intermediate_size (int): Dimensionality of the hidden layer.
        dropout (float): Dropout probability (not implemented here but can be added).

    Reference:
        Vaswani et al., "Attention Is All You Need" (2017)
    """
    def __init__(self, hidden_size=hidden_size, intermediate_size=intermediate_size, dropout=0.1):
        super(FFNLayer, self).__init__()

        # Not Gated Linear Unit
        self.fc_fn1 = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.activation = nn.GELU() # Model Transformer biasanya pakai SiLU cuma pakai GeLU cuma buat uji coba aja :D
        self.fc_fn2 = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)

    def forward(self, x):
        """
        Forward pass for FFN layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        x = self.fc_fn1(x)
        x = self.activation(x)
        x = self.fc_fn2(x)
        return x

class AttentionLayer(nn.Module):
    """
    Multi-head Self-Attention Layer as described in "Attention Is All You Need".

    This layer computes attention scores from input tensor and outputs
    the attended representations along with attention weights.

    Args:
        embed_dim (int): Total dimension of the model (hidden size).
        num_heads (int): Number of attention heads. embed_dim must be divisible by num_heads.

    Raises:
        AssertionError: If embed_dim is not divisible by num_heads.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.head_dim = embed_dim // num_heads # rumusnya kalo ga salah hidden_size dibagi num_heads :D

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=attn_bias)

        self.scale = self.head_dim ** -0.5

    def apply_rotary(self, q, k):
        """
        Apply rotary positional embeddings to q and k.

        Args:
            q, k: shape (batch_size, num_heads, seq_len, head_dim)

        Returns:
            Rotated q and k tensors with same shape.
        """
        seq_len = q.size(2)
        device = q.device
        dim = self.head_dim
        assert dim % 2 == 0, "head_dim must be even for rotary embeddings"

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        positions = torch.arange(seq_len, device=device).float()

        sinusoid_inp = torch.einsum('i,j->ij', positions, inv_freq)
        cos_emb = torch.cos(sinusoid_inp).unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,dim/2)
        sin_emb = torch.sin(sinusoid_inp).unsqueeze(0).unsqueeze(0)

        # reshape q,k to (..., dim/2, 2) for pairwise rotation
        q_ = q.view(*q.shape[:-1], dim // 2, 2)
        k_ = k.view(*k.shape[:-1], dim // 2, 2)

        # apply rotary: (x1,x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
        q_rot = torch.empty_like(q_)
        k_rot = torch.empty_like(k_)
        q_rot[..., 0] = q_[..., 0] * cos_emb - q_[..., 1] * sin_emb
        q_rot[..., 1] = q_[..., 0] * sin_emb + q_[..., 1] * cos_emb
        k_rot[..., 0] = k_[..., 0] * cos_emb - k_[..., 1] * sin_emb
        k_rot[..., 1] = k_[..., 0] * sin_emb + k_[..., 1] * cos_emb

        return q_rot.view_as(q), k_rot.view_as(k)

    def forward(self, x, mask=None):
        """
        Forward pass for multi-head self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, 1, 1, seq_len) or broadcastable,
                                           with 0s in positions to mask out.

        Returns:
            tuple:
                - output (torch.Tensor): Attention output tensor of shape (batch_size, seq_len, embed_dim).
                - attn_weights (torch.Tensor): Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size, seq_len, embed_dim = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K = self.apply_rotary(Q, K)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        output = self.out_proj(attn_output)

        return output, attn_weights


class RMSNorm(nn.Module):
    """
    RMSNorm layer implements Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm normalizes inputs based on their root mean square
    without subtracting the mean (no centering).

    Args:
        dim (int): Feature dimension of the input tensor.
        eps (float): Small epsilon to avoid division by zero.

    Reference:
        Zhang & Sennrich (2019) - https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim)

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input
        """
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        x_normed = x / (rms + self.eps)
        return x_normed * self.scale

class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer based on Attention Is All You Need.

    Structure:
        - Masked Multi-head Self-Attention + Residual + RMSNorm
        - Feedforward Network + Residual + RMSNorm

    Args:
        hidden_size (int): Embedding dimension.
        intermediate_size (int): Hidden dimension of FFN.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability (optional, not implemented here).

    Returns:
        output (torch.Tensor): (batch_size, seq_len, hidden_size)
        attn_weights (torch.Tensor): (batch_size, num_heads, seq_len, seq_len)
    """
    def __init__(self, hidden_size=hidden_size, intermediate_size=intermediate_size, num_heads=num_heads):
        super().__init__()
        self.self_attn = AttentionLayer(embed_dim=hidden_size, num_heads=num_heads)
        self.rms_norm1 = RMSNorm(hidden_size)
        self.ffn = FFNLayer(hidden_size, intermediate_size)
        self.rms_norm2 = RMSNorm(hidden_size)

    def forward(self, x, attn_mask=None):
        """
        Forward pass of DecoderLayer.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, hidden_size)
            attn_mask (torch.Tensor, optional): Attention mask for causal masking
                Shape expected to be broadcastable to (batch_size, num_heads, seq_len, seq_len)

        Returns:
            tuple:
                - output (torch.Tensor): Decoder output of shape (batch_size, seq_len, hidden_size)
                - attn_weights (torch.Tensor): Attention weights from self-attention
        """
        # Self-attention + residual connection + normalization
        attn_output, attn_weights = self.self_attn(x, mask=attn_mask)
        x = x + attn_output
        x = self.rms_norm1(x)

        # Feedforward + residual + normalization
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.rms_norm2(x)

        return x, attn_weights

class TRBlock(nn.Module):
    """
    TRBlock is a Transformer block, like the brain of a language model.

    It does these steps:
    1. Turns token IDs into vectors (embedding).
    2. Adds rotary position info (so it knows word order).
    3. Passes the vectors through 27 Transformer decoder layers.
    4. Normalizes the result (so values stay in a good range).
    5. Returns:
       - The final output.
       - All outputs from each layer (optional).
       - All attention scores (optional).
       - Place for cache if you want to make it faster later.

    Args:
        vocab_size (int): How many tokens the model understands (vocabulary size).
        hidden_size (int): Size of each hidden layer.
        intermediate_size (int): Size of the inner feedforward layer.
        num_heads (int): How many attention heads to use.
        max_seq_len (int): Max number of tokens in one sentence or input.
    """
    def __init__(self, vocab_size=vocab_size, hidden_size=hidden_size, intermediate_size=intermediate_size,
                 num_heads=num_heads, max_seq_len=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_size=hidden_size, intermediate_size=intermediate_size, num_heads=num_heads)
            for _ in range(num_hidden_layer)
        ])
        self.norm = RMSNorm(hidden_size)
        self.rotary = RotaryEmbedding(dim=hidden_size // num_heads, max_seq_len=max_seq_len)

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True, output_attentions=True):
        """
        Runs the Transformer block.

        Args:
            input_ids (Tensor): Token IDs with shape (batch, seq_len).
            attn_mask (Tensor, optional): A mask to block attention (like preventing cheating).
            output_hidden_states (bool): If True, gives back all layer outputs.
            output_attentions (bool): If True, gives back all attention maps.

        Returns:
            dict with:
                - last_hidden_state (Tensor): Final output from the last layer.
                - past_key_values (None): For caching (not used yet).
                - hidden_states (List[Tensor] or None): Outputs from each layer.
                - attentions (List[Tensor] or None): Attention scores from each layer.
        """
        hidden_states = []
        all_self_attentions = []

        x = self.embedding(input_ids)

        x = self.rotary(x)

        if output_hidden_states:
            hidden_states.append(x)

        for layer in self.decoder_layers:
            x, attn_weights = layer(x, attn_mask=attn_mask)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                all_self_attentions.append(attn_weights)

        x = self.norm(x)

        # x = self.rotary(x) # TAKUT ERROR xD

        return {
            "last_hidden_state": x,
            "past_key_values": None,
            "hidden_states": hidden_states if output_hidden_states else None,
            "attentions": all_self_attentions if output_attentions else None
        }

class TRForCausalLM(nn.Module):
    """
    TRForCausalLM is a Transformer-based language model for causal language modeling tasks.

    This module wraps the TRBlock backbone and adds a language modeling head (`lm_head`)
    to predict the next token in a sequence.

    Structure:
        - TRBlock: Transformer backbone with rotary embeddings and multiple decoder layers.
        - lm_head: Linear projection layer to map hidden states to vocabulary logits.

    Args:
        vocab_size (int): Size of the tokenizer vocabulary.
        hidden_size (int): Dimensionality of the model's hidden states.
        intermediate_size (int): Size of the feedforward hidden layer.
        num_heads (int): Number of attention heads in the model.
        max_seq_len (int): Maximum sequence length the model supports.
    """
    def __init__(self, vocab_size=vocab_size, hidden_size=hidden_size, intermediate_size=intermediate_size,
                 num_heads=num_heads, max_seq_len=max_seq_len):
        super().__init__()
        self.model = TRBlock(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            max_seq_len=max_seq_len
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None,
                output_hidden_states=False, output_attentions=False):
        """
        Forward pass for the TRForCausalLM model.

        Args:
            input_ids (torch.LongTensor): Tensor of shape (batch_size, seq_len) containing input token IDs.
            attn_mask (torch.Tensor, optional): Attention mask for masking out padding tokens or future tokens (for causal masking).
            labels (torch.LongTensor, optional): Target token IDs for computing the language modeling loss.
            output_hidden_states (bool, optional): If True, returns the hidden states from all layers.
            output_attentions (bool, optional): If True, returns attention weights from all layers.

        Returns:
            dict:
                - logits (torch.FloatTensor): Prediction scores of shape (batch_size, seq_len, vocab_size).
                - loss (torch.FloatTensor, optional): Cross-entropy loss if `labels` are provided.
                - hidden_states (List[torch.FloatTensor], optional): List of hidden states from each layer (if requested).
                - attentions (List[torch.FloatTensor], optional): List of attention maps from each layer (if requested).
        """
        attn_mask = attention_mask # literally sama untuk penyesuaian attn mask aja :D

        outputs = self.model(
            input_ids=input_ids,
            attn_mask=attn_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        logits = self.lm_head(outputs["last_hidden_state"])

        loss = None
        if labels is not None:
            # Shift for causal LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": outputs["hidden_states"] if output_hidden_states else None,
            "attentions": outputs["attentions"] if output_attentions else None
        }
