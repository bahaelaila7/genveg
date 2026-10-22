"""Learned neighbourhood embedding for the stand segmentation (user, 2026-10-22: Euclidean SLIC on the raw, heterogeneous feature stack suffers from
dimensionality and mixed scales; use a generative model of the ~90 m neighbourhood and segment its embedding).

Importance-weighted autoencoder (IWAE, Burda et al. 2016) of 3 x 3 pixel neighbourhoods (90 m):
  inputs per pixel  AlphaEarth 2025 (64, unit-norm embedding), AlphaEarth 2022 (64), TreeMap attributes (6, standardised; only on unharvested TreeMap
                    forest), soils (6, standardised) — missing groups are zeroed and flagged by 4 availability channels -> 144 channels
  encoder q(z|x)    Conv 3x3 (144 -> H) -> GELU -> Linear (H -> H) -> GELU -> Linear (H -> 2D): diagonal Gaussian over z in R^D
  prior             N(0, I): the posterior means live in an isotropic, unit-scaled space -> Euclidean distance is meaningful for SLIC / Ward
  decoder p(x|z)    MLP (D -> H -> H) -> for each of the 9 positions and 140 features a mean and a log-variance (heteroscedastic Gaussian: every
                    feature's scale and noise are learned, so no hand-set group weights); the likelihood counts only the available groups
  objective         IWAE bound log (1/K) sum_k p(x|z_k) p(z_k) / q(z_k|x), K importance samples
The encoder is fully convolutional: a whole tile is encoded at once (posterior mean per pixel, replicate padding at the tile edge)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
GROUPS = [('a25', 64), ('a22', 64), ('tm', 6), ('soil', 6)]
NF = sum(k for _, k in GROUPS); NG = len(GROUPS); CIN = NF + NG

def pixel_inputs(raw, fit):
    """(H, W, CIN) float32 model input from Featurizer.raw output + feature_fit (TreeMap / soil standardisation) and (H, W, NG) availability."""
    a25, a22 = raw['a25'], raw['a22']
    tm = (raw['tma'] - fit['tm_mean']) / fit['tm_sd']; so = (raw['soil'] - fit['soil_mean']) / fit['soil_sd']
    vals = [a25, a22, tm, so]; avail = np.stack([np.isfinite(v).all(-1) for v in vals], -1)
    X = np.concatenate([np.where(avail[..., [g]], np.nan_to_num(v), 0.0) for g, v in enumerate(vals)] + [avail.astype(np.float32)], -1)
    return X.astype(np.float32), avail

class IWAE(nn.Module):
    def __init__(self, D=12, H=128):
        super().__init__(); self.D = D
        self.conv = nn.Conv2d(CIN, H, 3); self.e1 = nn.Linear(H, H); self.e2 = nn.Linear(H, 2 * D)
        self.d1 = nn.Linear(D, H); self.d2 = nn.Linear(H, 2 * H); self.d3 = nn.Linear(2 * H, 9 * NF * 2)
    def encode_patches(self, x):                       # x: (B, CIN, 3, 3) -> mu, logvar (B, D)
        h = F.gelu(self.conv(x).flatten(1)); h = F.gelu(self.e1(h)); o = self.e2(h); return o[:, :self.D], o[:, self.D:].clamp(-10, 6)
    def encode_map(self, x):                           # x: (1, CIN, H, W) -> posterior mean (H, W, D)
        h = F.gelu(self.conv(F.pad(x, (1, 1, 1, 1), mode='replicate')))       # (1, H, H, W)
        h = h[0].permute(1, 2, 0); h = F.gelu(self.e1(h)); return self.e2(h)[..., :self.D]
    def decode(self, z):                               # z: (..., D) -> mean, logvar (..., 9, NF)
        h = F.gelu(self.d1(z)); h = F.gelu(self.d2(h)); o = self.d3(h).view(*z.shape[:-1], 9, NF, 2); return o[..., 0], o[..., 1].clamp(-8, 6)
    def iwae_loss(self, x, K=8):
        """-IWAE bound per patch (mean over the batch). Likelihood on the 9 positions x available groups only."""
        B = x.shape[0]; mu, lv = self.encode_patches(x)
        tgt = x[:, :NF].permute(0, 2, 3, 1).reshape(B, 9, NF)                                  # (B, 9, NF)
        av = x[:, NF:].permute(0, 2, 3, 1).reshape(B, 9, NG)
        fmask = torch.cat([av[..., [g]].expand(B, 9, k) for g, (_, k) in enumerate(GROUPS)], -1)  # (B, 9, NF)
        std = torch.exp(0.5 * lv); eps = torch.randn(K, B, self.D); z = mu + std * eps          # (K, B, D)
        m, l = self.decode(z)                                                                  # (K, B, 9, NF)
        ll = (-0.5 * (l + (tgt - m) ** 2 / torch.exp(l) + np.log(2 * np.pi)) * fmask).sum((-1, -2))
        lp = (-0.5 * (z ** 2 + np.log(2 * np.pi))).sum(-1); lq = (-0.5 * (eps ** 2 + lv + np.log(2 * np.pi))).sum(-1)
        lw = ll + lp - lq
        return -(torch.logsumexp(lw, 0) - np.log(K)).mean(), fmask.sum((-1, -2)).mean()

def load(path):
    ck = torch.load(path, weights_only=False); m = IWAE(ck['D'], ck['H']); m.load_state_dict(ck['state']); m.eval(); return m, ck

def encode_tile(model, raw, fit):
    """Posterior-mean embedding (H, W, D) of a window's Featurizer.raw output."""
    X, _ = pixel_inputs(raw, fit)
    with torch.no_grad():
        return model.encode_map(torch.from_numpy(np.ascontiguousarray(X.transpose(2, 0, 1)))[None]).numpy().astype(np.float32)
