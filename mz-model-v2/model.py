from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig


class MzModelConfig(PretrainedConfig):
    """Configuration for MzModelV2."""

    model_type = "mz_model_v2"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        max_position: int = 512,
        intermediate_size: int = 2048,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        # Accept the standard Transformers name as an alias for the project's
        # existing max_position setting.
        max_position = kwargs.pop("max_position_embeddings", max_position)

        # Keep the input embedding and output projection weights tied.
        kwargs["tie_word_embeddings"] = True
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_position = max_position
        self.max_position_embeddings = max_position
        self.intermediate_size = intermediate_size
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range

        if self.hidden_size <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_size and num_heads must be positive.")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if (self.hidden_size // self.num_heads) % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")
        if self.num_layers <= 0 or self.vocab_size <= 0:
            raise ValueError("num_layers and vocab_size must be positive.")
        if self.max_position <= 0 or self.intermediate_size <= 0:
            raise ValueError("max_position and intermediate_size must be positive.")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("rope_theta and rms_norm_eps must be positive.")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in the range [0, 1).")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return self.weight.to(dtype=input_dtype) * x_norm.to(dtype=input_dtype)


class RotaryEmbedding(nn.Module):
    """Computes RoPE cosine and sine values for a batch of position IDs."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate frequencies in FP32 for numerical stability.
        device = position_ids.device
        inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                / self.head_dim
            )
        )
        frequencies = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cos = embedding.cos().unsqueeze(1)
        sin = embedding.sin().unsqueeze(1)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    """Multi-head causal self-attention using PyTorch scaled dot-product attention."""

    def __init__(self, config: MzModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # PyTorch selects an available optimized SDPA backend automatically.
        # On supported CUDA hardware this can use FlashAttention or another fused backend.
        dropout_p = self.attention_dropout if self.training else 0.0
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(context)


class SwiGLU(nn.Module):
    """Gated feed-forward network with the SwiGLU activation."""

    def __init__(self, config: MzModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.down_proj(gate * value)


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer block with RoPE attention and SwiGLU."""

    def __init__(self, config: MzModelConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Attention(config)
        self.post_attention_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(self.input_norm(x), cos, sin)
        x = x + self.mlp(self.post_attention_norm(x))
        return x


class MzModelV2(PreTrainedModel):
    """Decoder-only language model using RoPE, RMSNorm, SwiGLU and SDPA."""

    config_class = MzModelConfig
    _tied_weights_keys = {"lm_head.weight": "token_embeddings.weight"}

    def __init__(self, config: MzModelConfig):
        super().__init__(config)

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.hidden_size // config.num_heads,
            theta=config.rope_theta,
        )

        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

        # Explicitly tie weights after initialization.
        self.lm_head.weight = self.token_embeddings.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model parameters using the configured normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.token_embeddings = value
        self.lm_head.weight = self.token_embeddings.weight

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value
        self.lm_head.weight = self.token_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length].")

        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            raise ValueError("input_ids must contain at least one token.")
        if seq_len > self.config.max_position:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum supported length "
                f"{self.config.max_position}."
            )

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=input_ids.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)
        elif position_ids.ndim == 1:
            if position_ids.shape[0] != seq_len:
                raise ValueError("1D position_ids must have sequence_length elements.")
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must have shape [sequence_length] or match input_ids.")

        cos, sin = self.rotary_emb(position_ids)
        x = self.token_embeddings(input_ids)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids.")
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss


__all__ = [
    "MzModelConfig",
    "MzModelV2",
    "RMSNorm",
    "RotaryEmbedding",
    "Attention",
    "SwiGLU",
    "TransformerBlock",
]
