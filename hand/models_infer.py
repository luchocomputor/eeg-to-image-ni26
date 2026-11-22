# --- Réseaux utilisés seulement à l'inférence (20/11), recopiés dans submission.py après models_v2.py ---


class SubjectEmbed(nn.Module):
    """Empreinte de sujet : EEG d'un essai (B, 32, 120) en référence moyenne -> vecteur unitaire. Entraîné à
    reconnaître les sujets d'Alljoined-1.6M ; à l'inférence, la ressemblance des empreintes dans le lot courant sert à
    retrouver les essais du même participant, pour estimer le décalage de prédiction propre à ce participant
    (statistiques du lot courant, permises par la FAQ ; rien n'est gardé d'un lot à l'autre)."""

    def __init__(self, n_chans=32, n_cls=16, Fm=32, K=35, S=5, d=64, drop=0.25):
        super().__init__()
        self.Fm, self.K, self.S = Fm, K, S
        self.temporal = nn.Linear(K, Fm)
        self.bn1 = nn.BatchNorm1d(Fm)
        self.spatial = nn.Linear(n_chans * Fm, 2 * Fm)
        self.bn2 = nn.BatchNorm1d(2 * Fm)
        self.drop = nn.Dropout(drop)
        self.emb = nn.Linear(4 * Fm, d)
        self.cls = nn.Linear(d, n_cls)

    def embed(self, x):
        B, C, _ = x.shape
        h = self.temporal(x.unfold(2, self.K, self.S))
        Tp = h.shape[2]
        h = nn.functional.elu(self.bn1(h.reshape(-1, self.Fm))).reshape(B, C, Tp, self.Fm)
        h = self.spatial(h.permute(0, 2, 1, 3).reshape(B, Tp, C * self.Fm))
        h = nn.functional.elu(self.bn2(h.reshape(-1, 2 * self.Fm))).reshape(B, Tp, 2 * self.Fm)
        h = torch.cat([h.mean(1), h.std(1)], 1)
        return nn.functional.normalize(self.emb(self.drop(h)), dim=-1)

    def forward(self, x):
        return self.cls(self.embed(x)) * 16.0
