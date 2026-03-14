"""
Exercise 1: KV-Cache for Decoder-Only Transformers — Solutions
==============================================================

Complete, runnable solutions for all 7 exercises in ex1-kv-cache.html.
Run this file to verify all solutions pass their tests.
"""

import torch
import torch as t
from torch import nn
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Starter code (provided) ──────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(t.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return x / rms * self.scale


class Attention(nn.Module):
    def __init__(self, d=64, n_head=4):
        super().__init__()
        self.n_head = n_head
        self.d_head = d // n_head
        self.QKV = nn.Linear(d, 3 * d, bias=False)
        self.O = nn.Linear(d, d, bias=False)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.QKV(x).chunk(3, dim=-1)
        q = q.view(b, s, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.n_head, self.d_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        mask = t.triu(t.ones(s, s, device=x.device, dtype=t.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.O(out)


class MLP(nn.Module):
    def __init__(self, d=64, exp=4):
        super().__init__()
        self.up = nn.Linear(d, exp * d, bias=False)
        self.gate = nn.Linear(d, exp * d, bias=False)
        self.down = nn.Linear(exp * d, d, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down(self.up(x) * self.act(self.gate(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d=64, n_head=4, exp=4):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, n_head)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, exp)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size=256, n_ctx=128, d=64, n_head=4, n_layers=4, exp=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(n_ctx, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, n_head, exp) for _ in range(n_layers)])
        self.norm = RMSNorm(d)
        self.unembed = nn.Linear(d, vocab_size, bias=False)

    def forward(self, tokens):
        b, s = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb(t.arange(s, device=tokens.device))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.unembed(x)


# ── Exercise 1: Naive autoregressive generation ──────────────────────────────

@t.no_grad()
def generate_naive(model, prompt, max_new_tokens=50, temperature=1.0):
    """Generate tokens autoregressively without KV cache."""
    tokens = prompt.clone()
    for _ in range(max_new_tokens):
        logits = model(tokens)
        next_logits = logits[:, -1, :] / temperature
        next_token = t.multinomial(next_logits.softmax(dim=-1), 1)
        tokens = t.cat([tokens, next_token], dim=1)
    return tokens


# ── Exercise 2: Conceptual (no code) ─────────────────────────────────────────
# Answers:
# 1. K, V for positions 1..n-1 are redundant. Only position n's K, V, Q are new.
# 2. No — causal mask means positions 1..n-1 attention outputs are unchanged.
# 3. Minimum: compute Q,K,V for new token only, retrieve cached K,V,
#    compute attention, run MLP on new position only.


# ── Exercise 3: KVCache ──────────────────────────────────────────────────────

class KVCache:
    """Stores cached key-value pairs for each layer."""
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def update(self, layer_idx, new_keys, new_values):
        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (new_keys, new_values)
        else:
            prev_k, prev_v = self.cache[layer_idx]
            self.cache[layer_idx] = (
                t.cat([prev_k, new_keys], dim=2),
                t.cat([prev_v, new_values], dim=2),
            )
        return self.cache[layer_idx]


# ── Exercise 4: CachedAttention ──────────────────────────────────────────────

class CachedAttention(nn.Module):
    def __init__(self, d=64, n_head=4):
        super().__init__()
        self.n_head = n_head
        self.d_head = d // n_head
        self.QKV = nn.Linear(d, 3 * d, bias=False)
        self.O = nn.Linear(d, d, bias=False)

    def forward(self, x, kv_cache=None, layer_idx=None):
        b, s, d = x.shape
        q, k, v = self.QKV(x).chunk(3, dim=-1)
        q = q.view(b, s, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.n_head, self.d_head).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        s_k = k.shape[2]
        s_q = q.shape[2]

        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Only apply causal mask if processing more than one query token
        if s_q > 1:
            mask = t.triu(t.ones(s_q, s_k, device=x.device, dtype=t.bool), diagonal=s_k - s_q + 1)
            attn = attn.masked_fill(mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, s_q, d)
        return self.O(out)


# ── Exercise 5: CachedTransformer ────────────────────────────────────────────

class CachedTransformerBlock(nn.Module):
    def __init__(self, d=64, n_head=4, exp=4):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = CachedAttention(d, n_head)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, exp)

    def forward(self, x, kv_cache=None, layer_idx=None):
        x = x + self.attn(self.norm1(x), kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + self.mlp(self.norm2(x))
        return x


class CachedTransformer(nn.Module):
    def __init__(self, vocab_size=256, n_ctx=128, d=64, n_head=4, n_layers=4, exp=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(n_ctx, d)
        self.blocks = nn.ModuleList(
            [CachedTransformerBlock(d, n_head, exp) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d)
        self.unembed = nn.Linear(d, vocab_size, bias=False)
        self.n_layers = n_layers

    def forward(self, tokens, kv_cache=None):
        b, s = tokens.shape

        # Determine position offset from cache
        if kv_cache is not None and kv_cache.cache[0] is not None:
            offset = kv_cache.cache[0][0].shape[2]  # cached seq length
        else:
            offset = 0

        positions = t.arange(offset, offset + s, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(positions)

        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache, layer_idx=i)

        x = self.norm(x)
        return self.unembed(x)


# ── Exercise 6: Cached generation ────────────────────────────────────────────

@t.no_grad()
def generate_cached(model, prompt, max_new_tokens=50, temperature=1.0):
    """Generate tokens autoregressively WITH KV cache."""
    tokens = prompt.clone()
    cache = KVCache(n_layers=model.n_layers)

    # Prefill: process the entire prompt
    logits = model(tokens, kv_cache=cache)
    next_logits = logits[:, -1, :] / temperature
    next_token = t.multinomial(next_logits.softmax(dim=-1), 1)
    tokens = t.cat([tokens, next_token], dim=1)

    # Decode: one token at a time
    for _ in range(max_new_tokens - 1):
        logits = model(next_token, kv_cache=cache)
        next_logits = logits[:, -1, :] / temperature
        next_token = t.multinomial(next_logits.softmax(dim=-1), 1)
        tokens = t.cat([tokens, next_token], dim=1)

    return tokens


# ── Exercise 7: Verification and benchmark ───────────────────────────────────

def test_correctness_and_benchmark():
    """Verify correctness and benchmark naive vs cached generation."""
    # Step 1: Verify correctness
    model_naive = Transformer(vocab_size=256, n_ctx=128, d=64, n_head=4, n_layers=4).to(device)
    model_cached = CachedTransformer(vocab_size=256, n_ctx=128, d=64, n_head=4, n_layers=4).to(device)
    model_cached.load_state_dict(model_naive.state_dict())

    prompt = t.randint(0, 256, (1, 10), device=device)

    logits_naive = model_naive(prompt)
    cache = KVCache(n_layers=4)
    logits_cached = model_cached(prompt, kv_cache=cache)

    assert t.allclose(logits_naive, logits_cached, atol=1e-5), "Logits don't match!"
    print("  Prefill logits match!")

    # Check one more token
    next_tok = t.randint(0, 256, (1, 1), device=device)
    full_seq = t.cat([prompt, next_tok], dim=1)
    logits_naive_2 = model_naive(full_seq)[:, -1:]
    logits_cached_2 = model_cached(next_tok, kv_cache=cache)
    assert t.allclose(logits_naive_2, logits_cached_2, atol=1e-5), "Decode logits don't match!"
    print("  Decode logits match!")

    # Step 2: Benchmark
    prompt = t.randint(0, 256, (1, 10), device=device)
    n_tokens = 100

    if device == "cuda":
        t.cuda.synchronize()
    start = time.time()
    for _ in range(3):
        generate_naive(model_naive, prompt, max_new_tokens=n_tokens)
        if device == "cuda":
            t.cuda.synchronize()
    time_naive = (time.time() - start) / 3

    if device == "cuda":
        t.cuda.synchronize()
    start = time.time()
    for _ in range(3):
        generate_cached(model_cached, prompt, max_new_tokens=n_tokens)
        if device == "cuda":
            t.cuda.synchronize()
    time_cached = (time.time() - start) / 3

    print(f"  Naive: {time_naive:.3f}s | Cached: {time_cached:.3f}s | Speedup: {time_naive/time_cached:.1f}x")


# ── Run all tests ────────────────────────────────────────────────────────────

def run_all_tests():
    print("=" * 60)
    print("Exercise 1: KV-Cache Solutions — Tests")
    print("=" * 60)

    # Test starter code
    print("\nStarter code:")
    model = Transformer().to(device)
    tokens = t.randint(0, 256, (2, 32), device=device)
    logits = model(tokens)
    assert logits.shape == (2, 32, 256)
    print(f"  Input {tokens.shape} -> Output {logits.shape} OK")

    # Exercise 1: Naive generation
    print("\nExercise 1 — Naive generation:")
    prompt = t.randint(0, 256, (1, 5), device=device)
    output = generate_naive(model, prompt, max_new_tokens=20)
    assert output.shape == (1, 25), f"Expected (1, 25), got {output.shape}"
    print("  Naive generation works!")

    # Exercise 3: KVCache
    print("\nExercise 3 — KVCache:")
    cache = KVCache(n_layers=4)
    k = t.randn(2, 4, 5, 16)
    v = t.randn(2, 4, 5, 16)
    all_k, all_v = cache.update(0, k, v)
    assert all_k.shape == (2, 4, 5, 16)
    k2 = t.randn(2, 4, 1, 16)
    v2 = t.randn(2, 4, 1, 16)
    all_k, all_v = cache.update(0, k2, v2)
    assert all_k.shape == (2, 4, 6, 16), f"Expected seq_len=6, got {all_k.shape}"
    print("  KVCache works!")

    # Exercise 4: CachedAttention
    print("\nExercise 4 — CachedAttention:")
    attn = CachedAttention(d=64, n_head=4).to(device)
    x = t.randn(1, 10, 64, device=device)
    out_full = attn(x)
    cache = KVCache(n_layers=1)
    out_cached = attn(x, kv_cache=cache, layer_idx=0)
    assert t.allclose(out_full, out_cached, atol=1e-5), "Cached prompt processing should match uncached!"
    print("  Cached attention matches uncached for prompt processing!")
    x_new = t.randn(1, 1, 64, device=device)
    out_new = attn(x_new, kv_cache=cache, layer_idx=0)
    assert out_new.shape == (1, 1, 64), f"Expected (1,1,64), got {out_new.shape}"
    print("  Single-token cached forward works!")

    # Exercise 5: CachedTransformer
    print("\nExercise 5 — CachedTransformer:")
    model_cached = CachedTransformer().to(device)
    tokens_test = t.randint(0, 256, (1, 10), device=device)
    cache = KVCache(n_layers=4)
    logits = model_cached(tokens_test, kv_cache=cache)
    assert logits.shape == (1, 10, 256)
    logits_next = model_cached(t.randint(0, 256, (1, 1), device=device), kv_cache=cache)
    assert logits_next.shape == (1, 1, 256)
    print("  CachedTransformer works!")

    # Exercise 6: Cached generation
    print("\nExercise 6 — Cached generation:")
    model_cached = CachedTransformer().to(device)
    prompt = t.randint(0, 256, (1, 5), device=device)
    output = generate_cached(model_cached, prompt, max_new_tokens=20)
    assert output.shape == (1, 25), f"Expected (1, 25), got {output.shape}"
    print("  Cached generation works!")

    # Exercise 7: Correctness and benchmark
    print("\nExercise 7 — Correctness & Benchmark:")
    test_correctness_and_benchmark()

    print("\n" + "=" * 60)
    print("All Exercise 1 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
