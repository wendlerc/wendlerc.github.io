"""
Exercise 4: KV Caching for Frame-Autoregressive Inference — Solutions
=====================================================================

Complete, runnable solutions for all 5 exercises in ex4-far-kv-cache.html.
Builds on Exercise 3 components (included here for self-containedness).
"""

import torch
import torch as t
from torch import nn
import torch.nn.functional as F
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Shared components (from Exercise 2 & 3) ──────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d=32, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(t.ones((1, 1, d)))
        self.eps = eps

    def forward(self, x):
        rms = (x.pow(2).mean(dim=-1, keepdim=True)).sqrt() + self.eps
        return x / rms * self.scale


class MLP(nn.Module):
    def __init__(self, d=64, exp=2):
        super().__init__()
        self.up = nn.Linear(d, exp * d, bias=False)
        self.gate = nn.Linear(d, exp * d, bias=False)
        self.down = nn.Linear(exp * d, d, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down(self.up(x) * self.act(self.gate(x)))


class NumEmbedding(nn.Module):
    def __init__(self, n_max, d=32, C=500):
        super().__init__()
        thetas = C ** (-t.arange(0, d // 2) / (d // 2))
        thetas = t.arange(0, n_max)[:, None].float() @ thetas[None, :]
        self.register_buffer("E", t.cat([t.sin(thetas), t.cos(thetas)], dim=1))

    def forward(self, x):
        return self.E[x]


class VideoPatch(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, d=64):
        super().__init__()
        self.patch_size = patch_size
        self.d = d
        self.conv = nn.Conv2d(in_channels, d, kernel_size=5, padding=2, stride=patch_size)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * T, -1, self.d)
        x = x.reshape(B, T * x.shape[1], self.d)
        return x

    def n_patches_per_frame(self, h, w):
        return (h // self.patch_size) * (w // self.patch_size)


class VideoUnPatch(nn.Module):
    def __init__(self, patch_size=4, d=64, out_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.up = nn.Linear(d, patch_size ** 2 * out_channels)

    def forward(self, x, n_frames):
        B, S, d = x.shape
        n_patches_per_frame = S // n_frames
        x = x.reshape(B * n_frames, n_patches_per_frame, d)
        x = self.up(x)
        w = int(n_patches_per_frame ** 0.5)
        h = w
        ps = self.patch_size
        c = self.out_channels
        x = x.reshape(B * n_frames, h, w, c, ps, ps)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(B * n_frames, c, h * ps, w * ps)
        x = x.reshape(B, n_frames, c, h * ps, w * ps)
        return x


def make_block_causal_mask(n_frames, patches_per_frame, device="cpu"):
    total = n_frames * patches_per_frame
    frame_idx = t.arange(total, device=device) // patches_per_frame
    mask = frame_idx[None, :] > frame_idx[:, None]
    return mask


# ── Non-cached CausalDiT (from Ex3, for comparison) ──────────────────────────

class CausalVideoAttention(nn.Module):
    def __init__(self, d=64, n_head=4):
        super().__init__()
        self.n_head = n_head
        self.d = d
        self.d_head = d // n_head
        self.QKV = nn.Linear(d, 3 * d, bias=False)
        self.O = nn.Linear(d, d, bias=False)
        self.normq = RMSNorm(self.d_head)
        self.normk = RMSNorm(self.d_head)

    def forward(self, x, mask=None):
        b, s, d = x.shape
        q, k, v = self.QKV(x).chunk(3, dim=-1)
        q = q.reshape(b, s, self.n_head, self.d_head)
        k = k.reshape(b, s, self.n_head, self.d_head)
        v = v.reshape(b, s, self.n_head, self.d_head)
        q = self.normq(q)
        k = self.normk(k)
        attn = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
        if mask is not None:
            attn = attn.masked_fill(mask[None, None, :, :], float('-inf'))
        attn = attn.softmax(dim=-1)
        z = attn @ v.permute(0, 2, 1, 3)
        z = z.permute(0, 2, 1, 3).reshape(b, s, self.d)
        return self.O(z)


class CausalDiTBlock(nn.Module):
    def __init__(self, d=64, n_head=4, exp=2):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = CausalVideoAttention(d, n_head)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, exp)
        self.modulate = nn.Linear(d, 6 * d)

    def forward(self, x, c, mask=None):
        scale1, bias1, gate1, scale2, bias2, gate2 = self.modulate(c).chunk(6, dim=-1)
        residual = x
        x = self.norm1(x) * (1 + scale1) + bias1
        x = self.attn(x, mask=mask) * gate1
        x = residual + x
        residual = x
        x = self.norm2(x) * (1 + scale2) + bias2
        x = self.mlp(x) * gate2
        x = residual + x
        return x


class CausalDiT(nn.Module):
    """Non-cached version for comparison."""
    def __init__(self, h=24, w=24, n_actions=4, in_channels=3,
                 patch_size=4, n_blocks=8, d=64, n_head=4, exp=2, T=1000):
        super().__init__()
        self.T = T
        self.patches_per_frame = (h // patch_size) * (w // patch_size)
        self.patch = VideoPatch(patch_size, in_channels, d)
        self.unpatch = VideoUnPatch(patch_size, d, in_channels)
        self.pe = nn.Parameter(t.randn(1, self.patches_per_frame, d) * d ** -0.5)
        self.te = NumEmbedding(T, d)
        self.ae = nn.Embedding(n_actions, d)
        self.act = nn.SiLU()
        self.blocks = nn.ModuleList([CausalDiTBlock(d, n_head, exp) for _ in range(n_blocks)])
        self.norm = RMSNorm(d)
        self.mod_final = nn.Linear(d, 2 * d)
        self.n_blocks = n_blocks

    def forward(self, x, actions, ts):
        B, T_frames, C, H, W = x.shape
        P = self.patches_per_frame
        tokens = self.patch(x)
        pe = self.pe.repeat(1, T_frames, 1)
        tokens = tokens + pe
        ts_int = t.minimum((ts * self.T).long(), t.tensor(self.T - 1, device=ts.device))
        cond = self.act(self.te(ts_int) + self.ae(actions))
        cond = cond.repeat_interleave(P, dim=1)
        mask = make_block_causal_mask(T_frames, P, device=x.device)
        for block in self.blocks:
            tokens = block(tokens, cond, mask=mask)
        scale, bias = self.mod_final(cond).chunk(2, dim=-1)
        tokens = self.norm(tokens) * (1 + scale) + bias
        return self.unpatch(tokens, n_frames=T_frames)

    @property
    def device(self):
        return self.pe.device


@t.no_grad()
def sample_video(model, first_frame, actions, n_denoise_steps=30, cfg=0):
    """Non-cached sampling for comparison."""
    B = first_frame.shape[0]
    total_frames = actions.shape[1]
    C, H, W = first_frame.shape[2:]
    video = first_frame.clone()
    for frame_idx in range(1, total_frames):
        z = t.randn(B, 1, C, H, W, device=first_frame.device, dtype=first_frame.dtype)
        denoise_ts = t.linspace(1, 0, n_denoise_steps + 1, device=first_frame.device)
        denoise_ts = 3 * denoise_ts / (2 * denoise_ts + 1)
        for step_idx in range(n_denoise_steps):
            current_video = t.cat([video, z], dim=1)
            n_ctx = current_video.shape[1]
            ts = t.zeros(B, n_ctx, device=first_frame.device)
            ts[:, -1] = denoise_ts[step_idx]
            act = actions[:, :n_ctx]
            v_pred = model(current_video, act, ts)
            if cfg > 0:
                v_uncond = model(current_video, act * 0, ts)
                v_pred = v_uncond + cfg * (v_pred - v_uncond)
            dt = denoise_ts[step_idx] - denoise_ts[step_idx + 1]
            z = z + dt * v_pred[:, -1:]
        video = t.cat([video, z], dim=1)
    return video


# ── Exercise 1: VideoKVCache ─────────────────────────────────────────────────

class VideoKVCache:
    """KV Cache for frame-autoregressive video generation."""
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def append(self, layer_idx, keys, values):
        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (keys, values)
        else:
            prev_k, prev_v = self.cache[layer_idx]
            self.cache[layer_idx] = (
                t.cat([prev_k, keys], dim=2),
                t.cat([prev_v, values], dim=2),
            )

    def get_cached_kv(self, layer_idx):
        if self.cache[layer_idx] is None:
            return None, None
        return self.cache[layer_idx]

    def get_and_extend(self, layer_idx, new_keys, new_values):
        cached_k, cached_v = self.get_cached_kv(layer_idx)
        if cached_k is None:
            return new_keys, new_values
        return (
            t.cat([cached_k, new_keys], dim=2),
            t.cat([cached_v, new_values], dim=2),
        )

    @property
    def cached_seq_len(self):
        if self.cache[0] is None:
            return 0
        return self.cache[0][0].shape[2]


# ── Exercise 2: CachedVideoAttention ─────────────────────────────────────────

class CachedVideoAttention(nn.Module):
    def __init__(self, d=64, n_head=4):
        super().__init__()
        self.n_head = n_head
        self.d = d
        self.d_head = d // n_head
        self.QKV = nn.Linear(d, 3 * d, bias=False)
        self.O = nn.Linear(d, d, bias=False)
        self.normq = RMSNorm(self.d_head)
        self.normk = RMSNorm(self.d_head)

    def forward(self, x, mask=None, kv_cache=None, layer_idx=None, cache_mode=None):
        b, s, d = x.shape
        q, k, v = self.QKV(x).chunk(3, dim=-1)
        q = q.reshape(b, s, self.n_head, self.d_head)
        k = k.reshape(b, s, self.n_head, self.d_head)
        v = v.reshape(b, s, self.n_head, self.d_head)
        q = self.normq(q)
        k = self.normk(k)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if kv_cache is not None and cache_mode == "finalize":
            # Store in cache and attend to everything
            kv_cache.append(layer_idx, k, v)
            k_full, v_full = kv_cache.get_cached_kv(layer_idx)
            attn = q @ k_full.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            out = attn @ v_full

        elif kv_cache is not None and cache_mode == "denoise":
            # Extend with cache but don't store
            k_full, v_full = kv_cache.get_and_extend(layer_idx, k, v)
            attn = q @ k_full.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            out = attn @ v_full

        else:
            # Standard attention with optional mask
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(mask[None, None, :, :], float('-inf'))
            attn = attn.softmax(dim=-1)
            out = attn @ v

        out = out.permute(0, 2, 1, 3).reshape(b, s, self.d)
        return self.O(out)


# ── Exercise 3: CachedCausalDiT ──────────────────────────────────────────────

class CachedCausalDiTBlock(nn.Module):
    def __init__(self, d=64, n_head=4, exp=2):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = CachedVideoAttention(d, n_head)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, exp)
        self.modulate = nn.Linear(d, 6 * d)

    def forward(self, x, c, mask=None, kv_cache=None, layer_idx=None, cache_mode=None):
        scale1, bias1, gate1, scale2, bias2, gate2 = self.modulate(c).chunk(6, dim=-1)
        residual = x
        x = self.norm1(x) * (1 + scale1) + bias1
        x = self.attn(x, mask=mask, kv_cache=kv_cache,
                      layer_idx=layer_idx, cache_mode=cache_mode) * gate1
        x = residual + x
        residual = x
        x = self.norm2(x) * (1 + scale2) + bias2
        x = self.mlp(x) * gate2
        x = residual + x
        return x


class CachedCausalDiT(nn.Module):
    def __init__(self, h=24, w=24, n_actions=4, in_channels=3,
                 patch_size=4, n_blocks=8, d=64, n_head=4, exp=2, T=1000):
        super().__init__()
        self.T = T
        self.n_blocks = n_blocks
        self.patches_per_frame = (h // patch_size) * (w // patch_size)
        self.patch = VideoPatch(patch_size, in_channels, d)
        self.unpatch = VideoUnPatch(patch_size, d, in_channels)
        self.pe = nn.Parameter(t.randn(1, self.patches_per_frame, d) * d ** -0.5)
        self.te = NumEmbedding(T, d)
        self.ae = nn.Embedding(n_actions, d)
        self.act = nn.SiLU()
        self.blocks = nn.ModuleList(
            [CachedCausalDiTBlock(d, n_head, exp) for _ in range(n_blocks)]
        )
        self.norm = RMSNorm(d)
        self.mod_final = nn.Linear(d, 2 * d)

    def forward(self, x, actions, ts, kv_cache=None, cache_mode=None):
        B, T_frames, C, H, W = x.shape
        P = self.patches_per_frame

        tokens = self.patch(x)
        pe = self.pe.repeat(1, T_frames, 1)
        tokens = tokens + pe

        ts_int = t.minimum((ts * self.T).long(), t.tensor(self.T - 1, device=ts.device))
        cond = self.act(self.te(ts_int) + self.ae(actions))
        cond = cond.repeat_interleave(P, dim=1)

        mask = None
        if cache_mode is None and T_frames > 1:
            mask = make_block_causal_mask(T_frames, P, device=x.device)

        for i, block in enumerate(self.blocks):
            tokens = block(tokens, cond, mask=mask, kv_cache=kv_cache,
                          layer_idx=i, cache_mode=cache_mode)

        scale, bias = self.mod_final(cond).chunk(2, dim=-1)
        tokens = self.norm(tokens) * (1 + scale) + bias

        return self.unpatch(tokens, n_frames=T_frames)

    @property
    def device(self):
        return self.pe.device


# ── Exercise 4: Cached Video Sampling ─────────────────────────────────────────

@t.no_grad()
def sample_video_cached(model, first_frame, actions, n_denoise_steps=30, cfg=0):
    """Generate video with KV-cached frame-autoregressive sampling."""
    B = first_frame.shape[0]
    total_frames = actions.shape[1]
    C, H, W = first_frame.shape[2:]

    cache_cond = VideoKVCache(model.n_blocks)
    cache_uncond = VideoKVCache(model.n_blocks) if cfg > 0 else None

    video = first_frame.clone()

    # Finalize the first frame
    ts_zero = t.zeros(B, 1, device=first_frame.device)
    act_first = actions[:, :1]
    model(first_frame, act_first, ts_zero, kv_cache=cache_cond, cache_mode="finalize")
    if cfg > 0:
        model(first_frame, act_first * 0, ts_zero, kv_cache=cache_uncond, cache_mode="finalize")

    for frame_idx in range(1, total_frames):
        z = t.randn(B, 1, C, H, W, device=first_frame.device, dtype=first_frame.dtype)

        denoise_ts = t.linspace(1, 0, n_denoise_steps + 1, device=first_frame.device)
        denoise_ts = 3 * denoise_ts / (2 * denoise_ts + 1)

        act_frame = actions[:, frame_idx:frame_idx + 1]

        for step_idx in range(n_denoise_steps):
            ts_frame = denoise_ts[step_idx] * t.ones(B, 1, device=first_frame.device)

            v_pred = model(z, act_frame, ts_frame,
                          kv_cache=cache_cond, cache_mode="denoise")[:, :1]

            if cfg > 0:
                v_uncond = model(z, act_frame * 0, ts_frame,
                               kv_cache=cache_uncond, cache_mode="denoise")[:, :1]
                v_pred = v_uncond + cfg * (v_pred - v_uncond)

            dt = denoise_ts[step_idx] - denoise_ts[step_idx + 1]
            z = z + dt * v_pred

        # Finalize: store clean frame's K,V in cache
        model(z, act_frame, ts_zero, kv_cache=cache_cond, cache_mode="finalize")
        if cfg > 0:
            model(z, act_frame * 0, ts_zero, kv_cache=cache_uncond, cache_mode="finalize")

        video = t.cat([video, z], dim=1)

    return video


# ── Exercise 5: Correctness & Benchmark ──────────────────────────────────────

def verify_correctness(model_naive, model_cached):
    """Verify cached and uncached produce same results."""
    first_frame = t.randn(1, 1, 3, 24, 24, device=device)
    act = t.ones(1, 4, dtype=t.long, device=device)

    t.manual_seed(42)
    video_naive = sample_video(model_naive, first_frame, act, n_denoise_steps=3, cfg=0)

    t.manual_seed(42)
    video_cached = sample_video_cached(model_cached, first_frame, act, n_denoise_steps=3, cfg=0)

    max_diff = (video_naive - video_cached).abs().max().item()
    return max_diff, video_naive, video_cached


# ── Run all tests ────────────────────────────────────────────────────────────

def run_all_tests():
    print("=" * 60)
    print("Exercise 4: KV Caching for FAR Inference — Tests")
    print("=" * 60)

    # Exercise 1: VideoKVCache
    print("\nExercise 1 — VideoKVCache:")
    cache = VideoKVCache(n_layers=4)
    assert cache.cached_seq_len == 0
    k = t.randn(2, 4, 9, 16)  # batch=2, heads=4, patches=9, d_head=16
    v = t.randn(2, 4, 9, 16)
    cache.append(0, k, v)
    assert cache.cached_seq_len == 9
    k2 = t.randn(2, 4, 9, 16)
    v2 = t.randn(2, 4, 9, 16)
    cache.append(0, k2, v2)
    assert cache.cached_seq_len == 18
    # Test get_and_extend
    k3 = t.randn(2, 4, 9, 16)
    v3 = t.randn(2, 4, 9, 16)
    k_ext, v_ext = cache.get_and_extend(0, k3, v3)
    assert k_ext.shape == (2, 4, 27, 16)  # 18 cached + 9 new
    # Cache should NOT have been modified by get_and_extend
    assert cache.cached_seq_len == 18
    print("  VideoKVCache works!")

    # Exercise 2: CachedVideoAttention
    print("\nExercise 2 — CachedVideoAttention:")
    attn = CachedVideoAttention(d=64, n_head=4).to(device)
    # Test standard mode (no cache)
    x = t.randn(1, 36, 64, device=device)
    mask = make_block_causal_mask(4, 9, device=device)
    out_standard = attn(x, mask=mask)
    assert out_standard.shape == (1, 36, 64)
    print("  Standard mode works!")

    # Test finalize mode
    cache = VideoKVCache(n_layers=1)
    x_frame = t.randn(1, 9, 64, device=device)  # one frame
    out_finalize = attn(x_frame, kv_cache=cache, layer_idx=0, cache_mode="finalize")
    assert out_finalize.shape == (1, 9, 64)
    assert cache.cached_seq_len == 9
    print("  Finalize mode works!")

    # Test denoise mode
    x_denoise = t.randn(1, 9, 64, device=device)  # new frame being denoised
    out_denoise = attn(x_denoise, kv_cache=cache, layer_idx=0, cache_mode="denoise")
    assert out_denoise.shape == (1, 9, 64)
    # Cache should still be 9 (denoise doesn't add to cache)
    assert cache.cached_seq_len == 9
    print("  Denoise mode works!")

    # Exercise 3: CachedCausalDiT
    print("\nExercise 3 — CachedCausalDiT:")
    model = CachedCausalDiT(h=24, w=24, n_actions=4, in_channels=3,
                             patch_size=4, n_blocks=2, d=64, n_head=4).to(device)

    # Test normal (uncached) mode
    x = t.randn(2, 3, 3, 24, 24, device=device)
    actions = t.randint(0, 4, (2, 3), device=device)
    ts = t.rand(2, 3, device=device)
    out = model(x, actions, ts)
    assert out.shape == (2, 3, 3, 24, 24)
    print("  Normal forward pass works!")

    # Test cached mode
    cache = VideoKVCache(model.n_blocks)
    x_one = t.randn(1, 1, 3, 24, 24, device=device)
    act_one = t.ones(1, 1, dtype=t.long, device=device)
    ts_zero = t.zeros(1, 1, device=device)
    out_finalize = model(x_one, act_one, ts_zero, kv_cache=cache, cache_mode="finalize")
    assert out_finalize.shape == (1, 1, 3, 24, 24)
    assert cache.cached_seq_len == model.patches_per_frame
    print("  Cached finalize works!")

    x_denoise = t.randn(1, 1, 3, 24, 24, device=device)
    ts_half = 0.5 * t.ones(1, 1, device=device)
    out_denoise = model(x_denoise, act_one, ts_half, kv_cache=cache, cache_mode="denoise")
    assert out_denoise.shape == (1, 1, 3, 24, 24)
    assert cache.cached_seq_len == model.patches_per_frame  # unchanged
    print("  Cached denoise works!")

    # Exercise 4: Cached video sampling
    print("\nExercise 4 — Cached video sampling:")
    first_frame = t.randn(1, 1, 3, 24, 24, device=device)
    act_seq = t.randint(1, 4, (1, 4), device=device)
    generated = sample_video_cached(model, first_frame, act_seq, n_denoise_steps=3, cfg=0)
    assert generated.shape == (1, 4, 3, 24, 24), f"Got {generated.shape}"
    assert t.allclose(generated[:, 0], first_frame[:, 0]), "First frame should be preserved!"
    print(f"  Cached sampling works! Shape: {generated.shape}")

    # Exercise 5: Correctness check (cached vs uncached)
    print("\nExercise 5 — Correctness (cached vs uncached):")
    # Create both models with same weights
    model_naive = CausalDiT(h=24, w=24, n_actions=4, in_channels=3,
                             patch_size=4, n_blocks=2, d=64, n_head=4).to(device)
    model_cached = CachedCausalDiT(h=24, w=24, n_actions=4, in_channels=3,
                                    patch_size=4, n_blocks=2, d=64, n_head=4).to(device)
    model_cached.load_state_dict(model_naive.state_dict())

    max_diff, _, _ = verify_correctness(model_naive, model_cached)
    print(f"  Max difference: {max_diff:.6f}")
    assert max_diff < 1e-3, f"Outputs diverge too much: {max_diff}"
    print("  Cached and uncached outputs match!")

    # Quick benchmark
    print("\n  Benchmark (3 frames, 3 denoise steps):")
    first_frame = t.randn(1, 1, 3, 24, 24, device=device)
    act = t.ones(1, 4, dtype=t.long, device=device)

    if device == "cuda":
        t.cuda.synchronize()
    start = time.time()
    sample_video(model_naive, first_frame, act, n_denoise_steps=3, cfg=0)
    if device == "cuda":
        t.cuda.synchronize()
    time_naive = time.time() - start

    if device == "cuda":
        t.cuda.synchronize()
    start = time.time()
    sample_video_cached(model_cached, first_frame, act, n_denoise_steps=3, cfg=0)
    if device == "cuda":
        t.cuda.synchronize()
    time_cached = time.time() - start

    speedup = time_naive / time_cached if time_cached > 0 else float('inf')
    print(f"  Naive: {time_naive:.3f}s | Cached: {time_cached:.3f}s | Speedup: {speedup:.1f}x")

    print("\n" + "=" * 60)
    print("All Exercise 4 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
