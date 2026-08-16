"""Configuration loading: YAML file -> pydantic Settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key_env: Optional[str] = None
    models: list[ModelConfig] = []
    default_model: Optional[str] = None

    def get_api_key(self) -> Optional[str]:
        if self.api_key_env is None:
            return None
        return os.environ.get(self.api_key_env)

    def get_model(self, model_name: Optional[str] = None) -> ModelConfig:
        """Return the ModelConfig for the given model name, or the default."""
        target = model_name or self.default_model
        for m in self.models:
            if m.name == target:
                return m
        if self.models:
            return self.models[0]
        raise ValueError(f"No models configured for provider {self.name}")


class RoutingConfig(BaseModel):
    strategy: str = "round_robin"
    max_retries: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 5.0


class CacheConfig(BaseModel):
    enabled: bool = True
    model_name: str = "all-MiniLM-L6-v2"
    backend: str = "torch"  # torch | onnx (onnx = torch-free, for constrained deploys)
    model_dir: Optional[str] = None  # path to ONNX model dir (required when backend=onnx)
    similarity_threshold: float = 0.95
    ttl_seconds: int = 3600
    max_size: int = 1000


class AppConfig(BaseModel):
    providers: list[ProviderConfig] = []
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Load YAML config file and return an AppConfig instance."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
