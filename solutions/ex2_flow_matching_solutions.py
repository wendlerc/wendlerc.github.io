"""
Exercise 2: Flow Matching on MNIST — Solutions
===============================================

Complete, runnable solutions for all 12 exercises in ex2-flow-matching-mnist.html.
Exercises 11–12 (training loops) are provided as functions but not run automatically
since they require MNIST data and training time.
"""

import torch
import torch as t
from torch import nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Exercise 1: Patch (Image → Tokens) ───────────────────────────────────────

class Patch(nn.Module):
    """Convert images to patch token sequences using strided convolution."""
    def __init__(self, patch_size=4, in_channels=1, d=32):
        super().__init__()
        self.d = d
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, d, kernel_size=5, padding=2, stride=patch_size)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv(x)                           # (b, d, h//ps, w//ps)
        x = x.permute(0, 2, 3, 1)                  # (b, h//ps, w//ps, d)
        x = x.reshape(b, -1, self.d)               # (b, n_patches, d)
        return x


# ── Exercise 2: UnPatch (Tokens → Image) ─────────────────────────────────────

class UnPatch(nn.Module):
    """Convert patch token sequences back to images."""
    def __init__(self, patch_size=4, d=32, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.up = nn.Linear(d, patch_size ** 2 * out_channels)

    def forward(self, x):
        b, s, d = x.shape
        x = self.up(x)                                              # (b, s, ps*ps*c)
        w = int(s ** 0.5)
        h = w
        ps = self.patch_size
        c = self.out_channels
        x = x.reshape(b, h, w, c, ps, ps)
        x = x.permute(0, 3, 1, 4, 2, 5)                            # (b, c, h, ps, w, ps)
        x = x.reshape(b, c, h * ps, w * ps)                        # (b, c, H, W)
        return x


# ── Exercise 3: RMSNorm ──────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d=32, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(t.ones((1, 1, d)))
        self.eps = eps

    def forward(self, x):
        rms = (x.pow(2).mean(dim=-1, keepdim=True)).sqrt() + self.eps
        return x / rms * self.scale


# ── Exercise 4: Attention with QK-Norm ───────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, d=32, n_head=4):
        super().__init__()
        self.n_head = n_head
        self.d = d
        self.d_head = d // n_head
        self.QKV = nn.Linear(d, 3 * d, bias=False)
        self.O = nn.Linear(d, d, bias=False)
        self.normq = RMSNorm(self.d_head)
        self.normk = RMSNorm(self.d_head)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.QKV(x).chunk(3, dim=-1)
        q = q.reshape(b, s, self.n_head, self.d_head)
        k = k.reshape(b, s, self.n_head, self.d_head)
        v = v.reshape(b, s, self.n_head, self.d_head)
        q = self.normq(q)
        k = self.normk(k)
        # (b, n_head, s, d_head)
        attn = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
        attn = attn.softmax(dim=-1)
        z = attn @ v.permute(0, 2, 1, 3)
        z = z.permute(0, 2, 1, 3).reshape(b, s, self.d)
        return self.O(z)


# ── Exercise 5: Gated MLP ────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, d=32, exp=2):
        super().__init__()
        self.up = nn.Linear(d, exp * d, bias=False)
        self.gate = nn.Linear(d, exp * d, bias=False)
        self.down = nn.Linear(exp * d, d, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down(self.up(x) * self.act(self.gate(x)))


# ── Exercise 6: Sinusoidal Embeddings ────────────────────────────────────────

class NumEmbedding(nn.Module):
    """Sinusoidal embeddings for integer values (timesteps)."""
    def __init__(self, n_max, d=32, C=500):
        super().__init__()
        thetas = C ** (-t.arange(0, d // 2) / (d // 2))
        thetas = t.arange(0, n_max)[:, None].float() @ thetas[None, :]
        sins = t.sin(thetas)
        coss = t.cos(thetas)
        self.register_buffer("E", t.cat([sins, coss], dim=1))

    def forward(self, x):
        return self.E[x]


# ── Exercise 7: DiT Block with AdaLN ─────────────────────────────────────────

class DiTBlock(nn.Module):
    def __init__(self, d=32, n_head=4, exp=2):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, n_head)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, exp)
        self.modulate = nn.Linear(d, 6 * d)

    def forward(self, x, c):
        scale1, bias1, gate1, scale2, bias2, gate2 = self.modulate(c).chunk(6, dim=-1)

        residual = x
        x = self.norm1(x) * (1 + scale1[:, None, :]) + bias1[:, None, :]
        x = self.attn(x) * gate1[:, None, :]
        x = residual + x

        residual = x
        x = self.norm2(x) * (1 + scale2[:, None, :]) + bias2[:, None, :]
        x = self.mlp(x) * gate2[:, None, :]
        x = residual + x
        return x


# ── Exercise 8: Full DiT Model ───────────────────────────────────────────────

class DiT(nn.Module):
    def __init__(self, h=28, w=28, n_classes=10, in_channels=1,
                 patch_size=4, n_blocks=4, d=32, n_head=4, exp=2, T=1000):
        super().__init__()
        self.T = T
        self.patch = Patch(patch_size, in_channels, d)
        self.n_seq = (h // patch_size) * (w // patch_size)
        self.pe = nn.Parameter(t.randn(1, self.n_seq, d) * d ** -0.5)
        self.te = NumEmbedding(T, d)
        self.ce = nn.Embedding(n_classes, d)
        self.act = nn.SiLU()
        self.blocks = nn.ModuleList([DiTBlock(d, n_head, exp) for _ in range(n_blocks)])
        self.norm = RMSNorm(d)
        self.modulate = nn.Linear(d, 2 * d)
        self.unpatch = UnPatch(patch_size, d, in_channels)

    def forward(self, x, c, ts):
        ts_int = t.minimum((ts * self.T).to(t.int64), t.tensor(self.T - 1, device=ts.device))
        cond = self.act(self.te(ts_int) + self.ce(c))
        x = self.patch(x) + self.pe
        for block in self.blocks:
            x = block(x, cond)
        scale, bias = self.modulate(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale[:, None, :]) + bias[:, None, :]
        x = self.unpatch(x)
        return x

    @property
    def device(self):
        return self.pe.device

    @property
    def dtype(self):
        return self.pe.dtype


# ── Exercise 9: Flow Matching Loss ───────────────────────────────────────────

def flow_matching_loss(model, x, c):
    """Compute the rectified flow matching loss."""
    batch_size = x.shape[0]
    ts = t.rand(batch_size, device=x.device, dtype=x.dtype)
    z = t.randn_like(x)
    v_true = x - z                                       # velocity: noise -> data
    x_t = x - ts[:, None, None, None] * v_true           # noisy image
    v_pred = model(x_t, c, ts)
    loss = F.mse_loss(v_pred, v_true)
    return loss


# ── Exercise 10: Euler Sampling ──────────────────────────────────────────────

@t.no_grad()
def sample(model, z, y, n_steps=30, cfg=0):
    """Generate images from noise using Euler integration."""
    ts = t.linspace(1, 0, n_steps + 1, device=z.device, dtype=z.dtype)
    ts = 3 * ts / (2 * ts + 1)  # SD3 scheduler
    for idx in range(n_steps):
        t_batch = ts[idx] * t.ones(z.shape[0], dtype=z.dtype, device=z.device)
        v_pred = model(z, y, t_batch)
        if cfg > 0:
            v_uncond = model(z, y * 0, t_batch)
            v_pred = v_uncond + cfg * (v_pred - v_uncond)
        z = z + (ts[idx] - ts[idx + 1]) * v_pred
    return z


# ── Exercise 11: Training loop (unconditional) ──────────────────────────────

def train_unconditional(model, train_loader, n_epochs=10, lr=1e-3):
    """Train the DiT model without class conditioning."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            c = t.zeros(images.shape[0], dtype=t.long, device=device)

            loss = flow_matching_loss(model, images, c)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")


# ── Exercise 12: Training loop (class-conditional with CFG) ──────────────────

def train_class_conditional(model, train_loader, n_epochs=10, lr=1e-3, label_dropout=0.2):
    """Train with class labels and CFG dropout."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            c = labels + 1  # shift: 0 reserved for unconditional
            drop_mask = t.rand(c.shape[0], device=device) < label_dropout
            c[drop_mask] = 0

            loss = flow_matching_loss(model, images, c)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")


# ── Run all tests ────────────────────────────────────────────────────────────

def run_all_tests():
    print("=" * 60)
    print("Exercise 2: Flow Matching on MNIST — Tests")
    print("=" * 60)

    # Exercise 1: Patch
    print("\nExercise 1 — Patch:")
    patch = Patch(patch_size=4, in_channels=1, d=32)
    x = t.randn(2, 1, 28, 28)
    tokens = patch(x)
    assert tokens.shape == (2, 49, 32), f"Expected (2, 49, 32), got {tokens.shape}"
    print(f"  Image {x.shape} -> Tokens {tokens.shape} OK")

    # Exercise 2: UnPatch
    print("\nExercise 2 — UnPatch:")
    unpatch = UnPatch(patch_size=4, d=32, out_channels=1)
    reconstructed = unpatch(tokens)
    assert reconstructed.shape == (2, 1, 28, 28), f"Expected (2, 1, 28, 28), got {reconstructed.shape}"
    print(f"  Tokens {tokens.shape} -> Image {reconstructed.shape} OK")

    # Exercise 3: RMSNorm
    print("\nExercise 3 — RMSNorm:")
    norm = RMSNorm(32)
    x_norm = norm(t.randn(2, 10, 32))
    assert x_norm.shape == (2, 10, 32)
    print("  RMSNorm works!")

    # Exercise 4: Attention
    print("\nExercise 4 — Attention with QK-Norm:")
    attn = Attention(d=32, n_head=4)
    x_attn = attn(t.randn(2, 49, 32))
    assert x_attn.shape == (2, 49, 32)
    print(f"  Attention output shape: {x_attn.shape} OK")

    # Exercise 5: MLP
    print("\nExercise 5 — Gated MLP:")
    mlp = MLP(d=32, exp=2)
    x_mlp = mlp(t.randn(2, 49, 32))
    assert x_mlp.shape == (2, 49, 32)
    print(f"  MLP output shape: {x_mlp.shape} OK")

    # Exercise 6: NumEmbedding
    print("\nExercise 6 — NumEmbedding:")
    emb = NumEmbedding(1000, d=32)
    out = emb(t.tensor([0, 100, 500, 999]))
    assert out.shape == (4, 32)
    assert not t.allclose(out[0], out[1])
    print("  NumEmbedding works!")

    # Exercise 7: DiTBlock
    print("\nExercise 7 — DiTBlock:")
    block = DiTBlock(d=32, n_head=4, exp=2)
    x_in = t.randn(2, 49, 32)
    c_in = t.randn(2, 32)
    x_out = block(x_in, c_in)
    assert x_out.shape == (2, 49, 32)
    print(f"  DiTBlock output shape: {x_out.shape} OK")

    # Exercise 8: Full DiT
    print("\nExercise 8 — Full DiT:")
    model = DiT(h=28, w=28, n_classes=11, d=32, n_head=4, n_blocks=4).to(device)
    x = t.randn(2, 1, 28, 28, device=device)
    c = t.tensor([3, 7], device=device)
    ts = t.tensor([0.5, 0.8], device=device)
    v = model(x, c, ts)
    assert v.shape == (2, 1, 28, 28), f"Expected (2, 1, 28, 28), got {v.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  DiT output shape: {v.shape}, Parameters: {n_params:,} OK")

    # Exercise 9: Flow matching loss
    print("\nExercise 9 — Flow matching loss:")
    loss = flow_matching_loss(model, x, c)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0
    print(f"  Loss: {loss.item():.4f} OK")

    # Exercise 10: Euler sampling
    print("\nExercise 10 — Euler sampling:")
    z = t.randn(2, 1, 28, 28, device=device)
    y = t.tensor([3, 7], device=device)
    samples = sample(model, z, y, n_steps=5, cfg=0)
    assert samples.shape == (2, 1, 28, 28)
    # Test with CFG
    samples_cfg = sample(model, z, y, n_steps=5, cfg=2.0)
    assert samples_cfg.shape == (2, 1, 28, 28)
    print(f"  Sampling works! Shape: {samples.shape} OK")

    # Exercises 11-12: Training loops (just verify they're callable)
    print("\nExercises 11-12 — Training functions defined (require MNIST data to run)")

    print("\n" + "=" * 60)
    print("All Exercise 2 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
