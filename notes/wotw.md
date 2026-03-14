# Getting into frame-autoregressive video and world models

A reading list and collection of resources for building up to frame-autoregressive video generation and world models. The progression goes: autoregressive transformers, then flow matching for images, then flow matching for video, and finally combining both with KV caching for efficient frame-autoregressive generation.

## 1. Decoder-only transformer + KV caching

Build a decoder-only transformer from scratch and understand KV caching for efficient autoregressive generation.

* [ARENA Chapter 1.1: Transformer from Scratch](https://arena-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch) -- step-by-step implementation of a transformer
* [minGPT / nanoGPT](https://github.com/karpathy/minGPT) -- Karpathy's clean, minimal GPT implementations

## 2. MNIST & class-conditional MNIST using flow matching

Learn diffusion and flow matching by generating images on MNIST. The references below trace the evolution from the original diffusion formulation to modern rectified flow matching.

* [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) -- the original DDPM paper
* [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502) -- deterministic sampling; for a while everyone used this
* [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) -- computational trick: run diffusion in a learned latent space to make everything much faster
* [Scalable Diffusion Models with Transformers (mmDiT)](https://arxiv.org/abs/2212.09748) -- the transformer architecture and conditioning mechanism that everyone is using now
* [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) -- nowadays everyone is doing rectified flow matching
* [Rectified flow matching on MNIST](https://github.com/wendlerc/mnist) -- beautiful minimal implementation on MNIST
* [minRF: Rectified flow matching on CIFAR](https://github.com/cloneofsimo/minRF) -- minimal implementation on CIFAR

## 3. Pong using flow matching

Move from images to video: generate pong games using flow-matching-based video models.

* [WAN: Scalable Bidirectional Text-to-Video Generation](https://arxiv.org/abs/2503.20314) -- bidirectional text-to-video generation
* [Diffusion Forcing](https://arxiv.org/abs/2407.01392) -- bridges autoregressive and diffusion models for sequence generation
* [CausVid](https://arxiv.org/abs/2412.07772v1) -- causal video generation (advanced)
* [Self-Forcing](https://arxiv.org/abs/2506.08009) -- training autoregressive video models without teacher forcing (advanced)
* [Minimal diffusion forcing implementation](https://github.com/wendlerc/toy-wm/) -- a toy implementation to get started

## 4. KV caching for frame-autoregressive transformers

Combine KV caching (from step 1) with flow-matching video generation (from step 3) for efficient frame-autoregressive world models.
