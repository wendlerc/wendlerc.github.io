"""
Exercise 3: Frame-Autoregressive Pong — Solutions
==================================================

Complete, runnable solutions for all 9 exercises in ex3-far-pong.html.
Imports shared components from ex2 solutions, and adds video-specific modules.
Exercises 1, 7, 9 (dataset / training / visualization) are provided as functions
but not run automatically since they require the Pong dataset.
"""

import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Reuse components from Exercise 2 ─────────────────────────────────────────
# (We redefine them here so this file is self-contained)

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
        sins = t.sin(thetas)
        coss = t.cos(thetas)
        self.register_buffer("E", t.cat([sins, coss], dim=1))

    def forward(self, x):
        return self.E[x]


# ── Exercise 1: PongDataset ──────────────────────────────────────────────────

class PongDataset(Dataset):
    """Dataset of Pong video sequences."""
    def __init__(self, frames, actions, seq_len=17, fps=5):
        super().__init__()
        self.seq_len = seq_len
        ep_len = fps * 60 + 1  # 5 fps * 60 seconds + 1
        n_episodes = len(frames) // ep_len
        frames = frames[:n_episodes * ep_len].reshape(n_episodes, ep_len, *frames.shape[1:])
        actions = actions[:n_episodes * ep_len].reshape(n_episodes, ep_len)

        self.frames = torch.tensor(frames, dtype=torch.float32) / 127.5 - 1
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.n_episodes = n_episodes
        self.seqs_per_episode = ep_len - seq_len + 1

    def __len__(self):
        return self.n_episodes * self.seqs_per_episode

    def __getitem__(self, idx):
        ep_idx = idx // self.seqs_per_episode
        frame_idx = idx % self.seqs_per_episode
        frames = self.frames[ep_idx, frame_idx:frame_idx + self.seq_len]
        actions = self.actions[ep_idx, frame_idx:frame_idx + self.seq_len]
        return frames, actions


# ── Exercise 2: VideoPatch / VideoUnPatch ─────────────────────────────────────

class VideoPatch(nn.Module):
    """Patchify video frames into a single token sequence."""
    def __init__(self, patch_size=4, in_channels=3, d=64):
        super().__init__()
        self.patch_size = patch_size
        self.d = d
        self.conv = nn.Conv2d(in_channels, d, kernel_size=5, padding=2, stride=patch_size)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.conv(x)                    # (B*T, d, h, w)
        x = x.permute(0, 2, 3, 1)          # (B*T, h, w, d)
        x = x.reshape(B * T, -1, self.d)   # (B*T, n_patches, d)
        x = x.reshape(B, T * x.shape[1], self.d)  # (B, T*n_patches, d)
        return x

    def n_patches_per_frame(self, h, w):
        return (h // self.patch_size) * (w // self.patch_size)


class VideoUnPatch(nn.Module):
    """Convert token sequence back to video frames."""
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


# ── Exercise 3: Block-Causal Attention Mask ──────────────────────────────────

def make_block_causal_mask(n_frames, patches_per_frame, device="cpu"):
    """Create block-causal mask. True = blocked (masked out)."""
    total = n_frames * patches_per_frame
    frame_idx = t.arange(total, device=device) // patches_per_frame
    mask = frame_idx[None, :] > frame_idx[:, None]
    return mask


# ── Exercise 4: Causal Video Attention ────────────────────────────────────────

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


# ── Exercise 5: CausalDiT ────────────────────────────────────────────────────

class CausalDiTBlock(nn.Module):
    """DiT block with causal video attention."""
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
        self.in_channels = in_channels
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

        output = self.unpatch(tokens, n_frames=T_frames)
        return output

    @property
    def device(self):
        return self.pe.device


# ── Exercise 6: Diffusion Forcing Loss ───────────────────────────────────────

def diffusion_forcing_loss(model, frames, actions, action_dropout=0.2):
    """Compute diffusion forcing loss for video."""
    B, T_frames, C, H, W = frames.shape
    ts = t.rand(B, T_frames, device=frames.device, dtype=frames.dtype)
    z = t.randn_like(frames)
    v_true = frames - z
    x_t = frames - ts[:, :, None, None, None] * v_true

    actions = actions.clone()
    drop_mask = t.rand(B, T_frames, device=frames.device) < action_dropout
    actions[drop_mask] = 0

    v_pred = model(x_t, actions, ts)
    loss = F.mse_loss(v_pred, v_true)
    return loss


# ── Exercise 7: Training loop ────────────────────────────────────────────────

def train_pong_model(model, train_loader, n_steps=2500, lr=3e-4):
    """Train the Pong video model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    running_loss = 0
    train_iter = iter(train_loader)

    for step in range(n_steps):
        try:
            frames, actions = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            frames, actions = next(train_iter)

        frames = frames.to(device)
        actions = actions.to(device)

        loss = diffusion_forcing_loss(model, frames, actions)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        running_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{n_steps} | Loss: {running_loss / 100:.4f}")
            running_loss = 0


# ── Exercise 8: Video Sampling ────────────────────────────────────────────────

@t.no_grad()
def sample_video(model, first_frame, actions, n_denoise_steps=30, cfg=0):
    """Generate video autoregressively, one frame at a time."""
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


# ── Exercise 9: Visualization (requires matplotlib, not tested here) ──────────
# See the HTML file for the show_video helper and visualization code.


# ── Run all tests ────────────────────────────────────────────────────────────

def run_all_tests():
    print("=" * 60)
    print("Exercise 3: Frame-Autoregressive Pong — Tests")
    print("=" * 60)

    # Exercise 2: VideoPatch / VideoUnPatch
    print("\nExercise 2 — VideoPatch / VideoUnPatch:")
    vpatch = VideoPatch(patch_size=4, in_channels=3, d=64)
    vunpatch = VideoUnPatch(patch_size=4, d=64, out_channels=3)
    video = t.randn(2, 5, 3, 24, 24)
    tokens = vpatch(video)
    n_patches = vpatch.n_patches_per_frame(24, 24)
    assert tokens.shape == (2, 5 * n_patches, 64), f"Got {tokens.shape}"
    reconstructed = vunpatch(tokens, n_frames=5)
    assert reconstructed.shape == (2, 5, 3, 24, 24), f"Got {reconstructed.shape}"
    print(f"  Video {video.shape} -> Tokens {tokens.shape} -> Video {reconstructed.shape} OK")

    # Exercise 3: Block-causal mask
    print("\nExercise 3 — Block-causal mask:")
    mask = make_block_causal_mask(4, 9)
    assert mask.shape == (36, 36)
    # Frame 0 tokens (0-8) should attend to themselves only
    assert mask[0, 0] == False   # can attend
    assert mask[0, 9] == True    # blocked (future frame)
    # Frame 1 tokens (9-17) should attend to frames 0 and 1
    assert mask[9, 0] == False   # can attend to frame 0
    assert mask[9, 9] == False   # can attend to own frame
    assert mask[9, 18] == True   # blocked (future frame)
    print(f"  Mask shape: {mask.shape}, correctness verified OK")

    # Exercise 4: CausalVideoAttention
    print("\nExercise 4 — CausalVideoAttention:")
    attn = CausalVideoAttention(d=64, n_head=4)
    x = t.randn(2, 36, 64)
    mask = make_block_causal_mask(4, 9)
    out = attn(x, mask=mask)
    assert out.shape == (2, 36, 64)
    print(f"  Attention output: {out.shape} OK")

    # Exercise 5: CausalDiT
    print("\nExercise 5 — CausalDiT:")
    model = CausalDiT(h=24, w=24, n_actions=4, in_channels=3,
                       patch_size=4, n_blocks=2, d=64, n_head=4).to(device)
    x = t.randn(2, 5, 3, 24, 24, device=device)
    actions = t.randint(0, 4, (2, 5), device=device)
    ts = t.rand(2, 5, device=device)
    out = model(x, actions, ts)
    assert out.shape == (2, 5, 3, 24, 24), f"Got {out.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CausalDiT output: {out.shape}, Parameters: {n_params:,} OK")

    # Exercise 6: Diffusion forcing loss
    print("\nExercise 6 — Diffusion forcing loss:")
    loss = diffusion_forcing_loss(model, x, actions)
    assert loss.ndim == 0
    assert loss.item() > 0
    print(f"  Loss: {loss.item():.4f} OK")

    # Exercise 8: Video sampling (small test)
    print("\nExercise 8 — Video sampling:")
    first_frame = t.randn(1, 1, 3, 24, 24, device=device)
    act_seq = t.randint(1, 4, (1, 4), device=device)
    # Use very few denoise steps for speed
    generated = sample_video(model, first_frame, act_seq, n_denoise_steps=3, cfg=0)
    assert generated.shape == (1, 4, 3, 24, 24), f"Got {generated.shape}"
    print(f"  Generated video: {generated.shape} OK")

    # Verify first frame is preserved
    assert t.allclose(generated[:, 0], first_frame[:, 0]), "First frame should be preserved!"
    print("  First frame preserved correctly!")

    print("\n" + "=" * 60)
    print("All Exercise 3 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
