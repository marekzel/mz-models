# MzModels

A collection of small decoder-only Transformer language models built from scratch using **PyTorch** and **Hugging Face Transformers**.

The project explores different Transformer architectures, training techniques, and model design choices, with a focus on training language models on consumer hardware.

## Models

| Model | Architecture | Key components |
|---|---|---|
| [MzModelV1](https://huggingface.co/marekzel/mz-model-v1) | Classic Transformer | GELU, LayerNorm, learned positional embeddings |
| [MzModelV2](https://huggingface.co/marekzel/mz-model-v2) | Modern Transformer | SwiGLU, RMSNorm, RoPE, scaled dot-product attention |

### MzModelV1

The first version uses a traditional decoder-only Transformer architecture.

- 12 Transformer layers
- 12 attention heads
- Hidden size: 768
- Vocabulary size: 32,000
- Maximum context length: 512 tokens
- GELU activation function
- LayerNorm normalization
- Learned positional embeddings
- Tied input and output embeddings

### MzModelV2

The second version introduces several architectural improvements commonly used in modern language models.

- RoPE (Rotary Positional Embeddings) for positional information
- RMSNorm for normalization
- SwiGLU gated feed-forward network
- Bias-free linear projections
- Scaled Dot-Product Attention (SDPA), using optimized PyTorch attention backends where available
- Tied input and output embeddings

## Training

The models are trained from scratch, rather than fine-tuned from existing pretrained language models.

The project uses a two-stage training approach:

1. **Pretraining** – learning language patterns and general knowledge from text datasets.
2. **Supervised Fine-Tuning (SFT)** – improving conversational abilities and instruction following.

Training experiments are performed on consumer GPU hardware, demonstrating practical approaches to building small language models without requiring a large GPU cluster.

## Disclaimer

These models are experimental projects created for educational and research purposes. They are not intended to compete with large-scale production language models. Their primary purpose is to explore language model architecture, training, and experimentation on accessible hardware.
