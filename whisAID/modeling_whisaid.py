from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

import whisper


DEFAULT_PRETRAINED_CHECKPOINT = ""
DEFAULT_CHECKPOINT_FILENAME = "checkpoint-epoch=0006.ckpt"


class WhisAIDConfig(PretrainedConfig):
    model_type = "whisaid"

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_PRETRAINED_CHECKPOINT,
        checkpoint_repo_id: Optional[str] = None,
        checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
        checkpoint_revision: Optional[str] = None,
        n_accents: int = 13,
        n_speakers: int = 292,
        n_mels: int = 80,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_repo_id = checkpoint_repo_id
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_revision = checkpoint_revision
        self.n_accents = n_accents
        self.n_speakers = n_speakers
        self.n_mels = n_mels


@dataclass
class WhisAIDOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    features: torch.FloatTensor = None


class WhisAIDForAccentClassification(PreTrainedModel):
    config_class = WhisAIDConfig
    base_model_prefix = "whisaid"

    def __init__(self, config: WhisAIDConfig):
        super().__init__(config)
        checkpoint_path = self._resolve_checkpoint(config)
        self.whisaid = whisper.load_model(
            checkpoint_path,
            n_accents=config.n_accents,
            n_speakers=config.n_speakers,
        )

    @staticmethod
    def _resolve_checkpoint(config: WhisAIDConfig) -> str:
        if config.checkpoint_repo_id:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(
                repo_id=config.checkpoint_repo_id,
                filename=config.checkpoint_filename,
                revision=config.checkpoint_revision,
            )
        if config.checkpoint_path:
            return config.checkpoint_path
        raise ValueError(
            "Set checkpoint_repo_id for a Hugging Face checkpoint or "
            "checkpoint_path for a local checkpoint."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        config_keys = {
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "token",
            "trust_remote_code",
        }
        config_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in config_keys}
        overrides = {
            key: kwargs.pop(key)
            for key in list(kwargs)
            if key in {"checkpoint_path", "checkpoint_repo_id", "checkpoint_filename", "checkpoint_revision"}
        }
        if config is None:
            config = WhisAIDConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
        for key, value in overrides.items():
            setattr(config, key, value)
        return cls(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_features: bool = True,
        **kwargs,
    ) -> WhisAIDOutput:
        audio_features = self.whisaid.encoder(input_ids)
        features, logits = self.whisaid.acc_head(audio_features.mean(dim=1))
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return WhisAIDOutput(
            loss=loss,
            logits=logits,
            features=features if output_features else None,
        )


AutoConfig.register(WhisAIDConfig.model_type, WhisAIDConfig)
AutoModel.register(WhisAIDConfig, WhisAIDForAccentClassification)
