# --- Architectures v2, torch pur : copiées telles quelles dans submission.py ---
import math

import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


def proj_head(d_in, d_out=1536, drop=0.5):
    return nn.Sequential(nn.Linear(d_in, d_out),
                         Residual(nn.Sequential(nn.GELU(), nn.Linear(d_out, d_out), nn.Dropout(drop))),
                         nn.LayerNorm(d_out))


class TSConv(nn.Module):
    """NICE (Song et al. 2024) adapté à 120 Hz : conv temporelle 100 ms, pooling 200 ms au pas
    de 25 ms, filtre spatial sur les canaux, tête de projection résiduelle vers DINOv2 (1536).
    attn=True : attention entre canaux en entrée (ATM, Li et al. 2024)."""

    def __init__(self, n_chans=32, n_times=120, F=40, k=12, pool=24, stride=3, drop=0.5, attn=False):
        super().__init__()
        self.attn = None
        if attn:
            self.pos = nn.Parameter(torch.zeros(1, n_chans, n_times))
            self.attn = nn.TransformerEncoderLayer(n_times, nhead=4, dim_feedforward=256,
                                                   dropout=0.1, batch_first=True, norm_first=True)
        self.conv = nn.Sequential(
            nn.Conv2d(1, F, (1, k)), nn.AvgPool2d((1, pool), (1, stride)),
            nn.BatchNorm2d(F), nn.ELU(),
            nn.Conv2d(F, F, (n_chans, 1)), nn.BatchNorm2d(F), nn.ELU(), nn.Dropout(drop),
            nn.Conv2d(F, F, (1, 1)))
        t_out = (n_times - k + 1 - pool) // stride + 1
        self.head = proj_head(F * t_out, drop=drop)

    def forward(self, x):
        if self.attn is not None:
            x = self.attn(x + self.pos)
        return self.head(self.conv(x.unsqueeze(1)).flatten(1))


class TSConvFast(nn.Module):
    """Même famille que TSConv, écrite en produits matriciels (rapides sur MPS, ~10x) :
    conv temporelle + pooling moyen (deux opérations linéaires) fusionnés en un filtre libre de
    K échantillons au pas S ; filtre spatial = couche linéaire sur (canaux x filtres)."""

    def __init__(self, n_chans=32, n_times=120, F=40, K=35, S=3, drop=0.5, attn=False):
        super().__init__()
        self.F, self.K, self.S = F, K, S
        self.attn = None
        if attn:
            self.pos = nn.Parameter(torch.zeros(1, n_chans, n_times))
            self.attn = nn.TransformerEncoderLayer(n_times, nhead=4, dim_feedforward=256,
                                                   dropout=0.1, batch_first=True, norm_first=True)
        self.temporal = nn.Linear(K, F)
        self.bn1 = nn.BatchNorm1d(F)
        self.spatial = nn.Linear(n_chans * F, F)
        self.bn2 = nn.BatchNorm1d(F)
        self.drop = nn.Dropout(drop)
        self.mix = nn.Linear(F, F)
        t_out = (n_times - K) // S + 1
        self.head = proj_head(F * t_out, drop=drop)

    def forward(self, x):                                   # x (B, C, T)
        if self.attn is not None:
            x = self.attn(x + self.pos)
        B, C, _ = x.shape
        h = self.temporal(x.unfold(2, self.K, self.S))      # (B, C, T', F)
        Tp = h.shape[2]
        h = nn.functional.elu(self.bn1(h.reshape(-1, self.F))).reshape(B, C, Tp, self.F)
        h = self.spatial(h.permute(0, 2, 1, 3).reshape(B, Tp, C * self.F))   # (B, T', F)
        h = nn.functional.elu(self.bn2(h.reshape(-1, self.F))).reshape(B, Tp, self.F)
        h = self.mix(self.drop(h))
        return self.head(h.reshape(B, -1))


# --- REVE (Elouahidi et al. 2025), copie autonome du code braindecode 1.8.1 (mêmes noms de paramètres que
# braindecode.models.REVE, poids Hugging Face brain-bzh/reve-base) : torch seul, aucune dépendance ni téléchargement
# à l'inférence. Positions (x, y, z) des 32 électrodes Emotiv : banque de positions REVE.
REVE_POS32 = [
    (0.000401, -0.009167, 0.100244), (0.000376, 0.02739, 0.088668), (0.000231, 0.080771, 0.035417), (-0.029437, 0.083917, -0.00699),
    (-0.064466, 0.048035, 0.016921), (-0.027496, 0.056931, 0.060342), (-0.079592, -0.046551, 0.030949), (-0.063556, -0.047009, 0.065624),
    (-0.035513, -0.047292, 0.091315), (-0.02862, -0.080525, 0.075436), (-0.053007, -0.078788, 0.05594), (-0.067272, -0.076291, 0.028382),
    (-0.072434, -0.073453, -0.002487), (-0.05484, -0.097528, 0.002792), (-0.036511, -0.100853, 0.037167), (-0.029413, -0.112449, 0.008839),
    (0.000325, -0.081115, 0.082615), (0.000216, -0.102178, 0.050608), (0.000108, -0.114892, 0.014657), (0.029843, -0.112156, 0.0088),
    (0.036782, -0.100849, 0.036397), (0.055667, -0.097625, 0.00273), (0.073056, -0.073068, -0.00254), (0.067888, -0.075904, 0.028091),
    (0.055667, -0.07856, 0.056561), (0.03192, -0.080487, 0.076716), (0.038384, -0.047073, 0.090695), (0.066612, -0.046637, 0.06558),
    (0.083322, -0.046101, 0.031206), (0.029514, 0.057602, 0.05954), (0.067914, 0.04983, 0.016367), (0.029872, 0.084896, -0.00708),
]


class _GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return nn.functional.gelu(gates) * x


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class _FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(_RMSNorm(dim), nn.Linear(dim, hidden_dim * 2, bias=False), _GEGLU(),
                                 nn.Linear(hidden_dim, dim, bias=False))

    def forward(self, x):
        return self.net(x)


class _Attention(nn.Module):
    def __init__(self, dim, heads=8, head_dim=64):
        super().__init__()
        self.heads = heads
        self.norm = _RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, head_dim * heads * 3, bias=False)
        self.to_out = nn.Linear(head_dim * heads, dim, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q, k, v = (t.reshape(B, S, self.heads, -1).transpose(1, 2) for t in self.to_qkv(self.norm(x)).chunk(3, -1))
        out = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.to_out(out.transpose(1, 2).reshape(B, S, -1))


class _Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([_Attention(dim, heads, head_dim), _FeedForward(dim, mlp_dim)])
                                     for _ in range(depth)])
        self.ckpt = False         # recalcul des activations à la rétropropagation (REVE-large sur T4 15 Go)

    @staticmethod
    def _block(attn, ff, x):
        x = attn(x) + x
        return ff(x) + x

    def forward(self, x):
        for attn, ff in self.layers:
            if self.ckpt and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(self._block, attn, ff, x, use_reentrant=False)
            else:
                x = self._block(attn, ff, x)
        return x


def _fourier4d(pos, dim, freqs=4, increment_time=0.1, margin=0.4):
    pos = pos.clone()
    pos[:, :, -1] *= increment_time
    shape = list(pos.shape[:-1])
    fw = torch.arange(freqs).to(pos)
    fz = fw[:, None]
    fy = fz[:, None]
    fx = fy[:, None]
    width = 1 + 2 * margin
    pos = (pos + margin)[..., None, None, None, None, :]
    c = 2 * math.pi / width
    loc = pos[..., 0] * c * fx + pos[..., 1] * c * fy + pos[..., 2] * c * fz + pos[..., 3] * c * fw
    loc = loc.view(shape + [-1])[..., :dim // 2]
    return torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)


class ReveCore(nn.Module):
    """REVE-base, tête au choix : att = regroupement par attention (requête pré-entraînée) puis LayerNorm + Linear ;
    flat = jetons (canal x patch) concaténés puis LayerNorm + Linear (tête par défaut de braindecode)."""

    def __init__(self, n_out=1536, embed_dim=512, depth=22, heads=8, head_dim=64, mlp_ratio=2.66, patch=200,
                 overlap=20, freqs=4, head="att", n_tokens=32):
        super().__init__()
        self.head = head
        self.patch, self.step, self.embed_dim, self.freqs = patch, patch - overlap, embed_dim, freqs
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch, embed_dim))
        self.mlp4d = nn.Sequential(nn.Linear(4, embed_dim, bias=False), nn.GELU(), nn.LayerNorm(embed_dim))
        self.ln = nn.LayerNorm(embed_dim)
        self.transformer = _Transformer(embed_dim, depth, heads, head_dim, int(embed_dim * mlp_ratio))
        d = embed_dim * (n_tokens if head == "flat" else 1)
        self.final_layer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, eeg, pos):
        patches = eeg.unfold(2, self.patch, self.step)                    # (B, C, P, patch)
        B, C, P, _ = patches.shape
        t = torch.arange(P, device=pos.device).float().view(1, 1, P, 1).expand(B, C, P, 1)
        pos4 = torch.cat([pos.unsqueeze(2).repeat(1, 1, P, 1), t], -1).view(B, C * P, 4)
        x = self.to_patch_embedding(patches).reshape(B, C * P, -1)
        x = x + self.ln(_fourier4d(pos4, self.embed_dim, self.freqs) + self.mlp4d(pos4))
        x = self.transformer(x)
        if self.head == "flat":
            return self.final_layer(x.reshape(B, -1))
        att = (self.cls_query_token.expand(B, -1, -1) @ x.transpose(-1, -2) / self.embed_dim ** 0.5).softmax(-1)
        return self.final_layer((att @ x).squeeze(1))


class ReveRetrieval(nn.Module):
    """REVE-base affiné de bout en bout, la piste des meilleures équipes du classement. Entrée
    (B, 32, 120) en ligne de base et écrêtée : 120 -> 200 éch. (linéaire), z-score par essai et par canal,
    écrêtage +-15 (comme la sonde REVE officielle) ; float16 sur GPU. Poids pré-entraînés à l'entraînement :
    variable NI_REVE (model.safetensors de brain-bzh/reve-base)."""

    def __init__(self, n_chans=32, n_times=120, n_out=1536, norm="trial", head="att", size="base"):
        super().__init__()
        import os
        self.norm = norm          # trial : z-score par essai et par canal ; global : échelle de l'enregistrement gardée
        big = dict(embed_dim=1216, heads=19, freqs=5) if size == "large" else {}     # brain-bzh/reve-large (~390 M)
        self.reve = ReveCore(n_out, head=head, n_tokens=n_chans, **big)     # 200 éch. = 1 patch par canal
        self.register_buffer("pos", torch.tensor(REVE_POS32[:n_chans], dtype=torch.float32))
        ck = os.environ.get("NI_REVE")
        if ck:
            from safetensors.torch import load_file
            miss, unexp = self.reve.load_state_dict(load_file(ck), strict=False)
            assert not unexp and all(k.startswith("final_layer") for k in miss), (miss, unexp)

    def backbone_parameters(self):
        return [p for n, p in self.reve.named_parameters() if not n.startswith("final_layer")]

    def forward(self, x):
        x = nn.functional.interpolate(x.float(), size=200, mode="linear", align_corners=False)
        if self.norm == "trial":
            x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        x = x.clamp(-15, 15)    # global : signal déjà normalisé par canal et par enregistrement (RobustScaler officiel)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=x.is_cuda):
            out = self.reve(x, self.pos.expand(len(x), -1, -1))
        return out.float()


class QualityGate(nn.Module):
    """Porte de qualité par essai (20/11) : petit réseau de la famille TSConvFast, EEG (B, 32, 120) en référence
    moyenne -> logit « l'essai contient une réponse évoquée exploitable ». Entraîné à distinguer un essai réel de son
    substitut à phases aléatoires (même spectre, même covariance spatiale, plus de réponse calée sur l'image), de
    mélanges réel/substitut et d'artefacts simulés. La plateforme moyenne les sorties BRUTES des répétitions : la
    sortie de l'ensemble est multipliée par sigmoid(logit)^gamma, ce qui sous-pondère les essais sans signal."""

    def __init__(self, n_chans=32, n_times=120, Fm=24, K=35, S=3, drop=0.25):
        super().__init__()
        self.Fm, self.K, self.S = Fm, K, S
        self.temporal = nn.Linear(K, Fm)
        self.bn1 = nn.BatchNorm1d(Fm)
        self.spatial = nn.Linear(n_chans * Fm, Fm)
        self.bn2 = nn.BatchNorm1d(Fm)
        self.drop = nn.Dropout(drop)
        t_out = (n_times - K) // S + 1
        self.out = nn.Sequential(nn.Linear(Fm * t_out, 64), nn.ELU(), nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, x):                                   # x (B, C, T), référence moyenne déjà appliquée
        B, C, _ = x.shape
        h = self.temporal(x.unfold(2, self.K, self.S))
        Tp = h.shape[2]
        h = nn.functional.elu(self.bn1(h.reshape(-1, self.Fm))).reshape(B, C, Tp, self.Fm)
        h = self.spatial(h.permute(0, 2, 1, 3).reshape(B, Tp, C * self.Fm))
        h = nn.functional.elu(self.bn2(h.reshape(-1, self.Fm))).reshape(B, -1)
        return self.out(self.drop(h)).squeeze(1)


def build_net(arch, n_chans=32, n_times=120):
    if arch in ("reve", "reveg", "revef", "revel", "revelf"):
        return ReveRetrieval(n_chans=n_chans, n_times=n_times, norm="global" if arch == "reveg" else "trial",
                             head="flat" if arch.endswith("f") else "att",
                             size="large" if arch.startswith("revel") else "base")
    if arch == "eegnet":
        from braindecode.models import EEGNet
        return EEGNet(n_chans=n_chans, n_outputs=1536, n_times=n_times)
    if arch in ("tsconv", "atm"):
        return TSConv(n_chans=n_chans, n_times=n_times, attn=arch == "atm")
    if arch in ("fast", "fastatm"):
        return TSConvFast(n_chans=n_chans, n_times=n_times, attn=arch == "fastatm")
    if arch == "fastwide":
        return TSConvFast(n_chans=n_chans, n_times=n_times, F=80)
    raise ValueError(arch)
