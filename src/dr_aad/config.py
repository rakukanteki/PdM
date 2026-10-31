"""Explicit, serializable settings. Unknown keys fail rather than silently disappearing."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    duration: float = 10.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    patch_frames: int = 15
    train_stride: int = 5
    eval_stride: int = 3
    train_patches: int = 35
    eval_patches: int = 96
    top_db: float = 80.0
    normalization_floor: float = 1e-3
    noise_seed: int = 7123
    noise_mode: str = "shared"  # identical normalized noise for equal-length recordings


@dataclass
class ModelConfig:
    stem_channels: int = 64
    descriptor_hidden: int = 64
    se_reduction: int = 16
    dropout: float = 0.3


@dataclass
class TrainConfig:
    epochs: int = 300
    patience: int = 20
    patch_batch_size: int = 128
    eval_batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    min_learning_rate: float = 3e-6
    descriptor_weight: float = 0.10
    relational_weight: float = 0.15
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.4
    augmentation_std: float = 0.01
    frequency_mask: int = 15
    time_mask: int = 3
    mask_rounds: int = 2
    gradient_clip: float = 5.0
    auxiliary_ids: int = 2
    auxiliary_recordings_per_id: int = 2
    seeds: list[int] = field(default_factory=lambda: [42, 1337])
    workers: int = 0
    deterministic: bool = True


@dataclass
class ScoreConfig:
    variance_floor: float = 1e-4
    diagonal_ridge: float = 0.05
    false_alarm_fraction: float = 0.05
    review_fraction: float = 0.10
    max_fpr: float = 0.10


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    scoring: ScoreConfig = field(default_factory=ScoreConfig)

    @classmethod
    def from_dict(cls, values):
        constructors = {"audio": AudioConfig, "model": ModelConfig,
                        "training": TrainConfig, "scoring": ScoreConfig}
        unknown = set(values) - set(constructors)
        if unknown:
            raise ValueError(f"Unknown configuration sections: {unknown}")
        obj = cls(**{k: constructors[k](**v) for k, v in values.items()})
        obj.validate()
        return obj

    @classmethod
    def load(cls, path=None):
        return cls.from_dict(json.loads(Path(path).read_text()) if path else {})

    def to_dict(self):
        return asdict(self)

    def validate(self):
        a, t, s = self.audio, self.training, self.scoring
        if a.sample_rate != 16000 or a.n_mels != 128:
            raise ValueError("This descriptor/encoder implementation requires 16 kHz and 128 Mel bands.")
        for key in ("n_fft", "hop_length", "patch_frames", "train_stride", "eval_stride",
                    "train_patches", "eval_patches"):
            if getattr(a, key) <= 0:
                raise ValueError(f"audio.{key} must be positive")
        if a.duration <= 0 or a.noise_mode not in {"shared", "independent"}:
            raise ValueError("Invalid audio duration or noise_mode")
        if t.epochs < 1 or t.patience < 1 or min(t.patch_batch_size, t.eval_batch_size) < 1:
            raise ValueError("Epochs, patience, and batch sizes must be positive")
        if t.auxiliary_ids < 2 or t.auxiliary_recordings_per_id < 2:
            raise ValueError("Use at least two IDs and two distinct recordings per auxiliary ID")
        if t.descriptor_weight < 0 or t.relational_weight < 0:
            raise ValueError("Alignment loss coefficients must be nonnegative")
        if not t.seeds or len(t.seeds) != len(set(t.seeds)):
            raise ValueError("Provide at least one unique training seed")
        if not 0 < s.max_fpr <= 1 or not 0 < s.false_alarm_fraction < 1:
            raise ValueError("Invalid scoring fractions")
        if not 0 < s.review_fraction < 1 or s.variance_floor <= 0 or s.diagonal_ridge < 0:
            raise ValueError("Invalid scoring settings")
