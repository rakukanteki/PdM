"""Residual-SE acoustic encoder and separate descriptor fusion/reconstruction pathways."""
import torch
from torch import nn

VARIANTS = ("cnn", "fusion", "relational", "full")


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(channels, max(1, channels // reduction)),
                                    nn.ReLU(), nn.Linear(max(1, channels // reduction), channels),
                                    nn.Sigmoid())

    def forward(self, x):
        return x * self.layers(x.mean(dim=(2, 3)))[:, :, None, None]


class ResidualSE(nn.Module):
    def __init__(self, cin, cout, reduction):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(), nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            SqueezeExcitation(cout, reduction))
        self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=2, bias=False), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.residual(x) + self.skip(x))


class DRAAD(nn.Module):
    def __init__(self, num_ids, cfg, variant="full"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        c, h = cfg.stem_channels, cfg.descriptor_hidden
        self.embedding_dim = 8 * c
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
            ResidualSE(c, c * 2, cfg.se_reduction),
            ResidualSE(c * 2, c * 4, cfg.se_reduction),
            ResidualSE(c * 4, c * 8, cfg.se_reduction), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        if variant != "cnn":
            self.descriptor_branch = nn.Sequential(nn.Linear(11, h), nn.LayerNorm(h),
                                                   nn.ReLU(), nn.Linear(h, h))
            self.fusion = nn.Sequential(nn.Linear(self.embedding_dim + h, self.embedding_dim),
                                        nn.ReLU(), nn.LayerNorm(self.embedding_dim))
        else:
            self.descriptor_branch = self.fusion = None
        self.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(self.embedding_dim, num_ids))
        # Present only where optimized; no untrained decoder parameters in ablation counts.
        self.decoder = nn.Linear(self.embedding_dim, 11) if variant == "full" else None

    def forward(self, patches, descriptors):
        acoustic = self.encoder(patches)
        if self.fusion is None:
            fused = acoustic
        else:
            context = self.descriptor_branch(descriptors)
            fused = self.fusion(torch.cat([acoustic, context], dim=-1))
        return acoustic, fused, self.classifier(fused)

    def reconstruct(self, acoustic_recordings):
        if self.decoder is None:
            raise RuntimeError("This ablation has no descriptor reconstruction decoder")
        return self.decoder(acoustic_recordings)
