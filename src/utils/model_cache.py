"""
Model caching utilities for automatic model download and local storage.

Handles downloading models from HuggingFace Hub and caching them
locally to avoid repeated downloads.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Manager for caching HuggingFace models locally.

    Features:
    - Automatic download from HuggingFace Hub
    - Local caching to avoid repeated downloads
    - Organized storage with model ID as directory name
    """

    def __init__(self, cache_dir: Path, hf_token: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token
        logger.info("Model cache directory: %s", self.cache_dir.absolute())

    def get_local_path(self, model_id: str) -> Path:
        """Convert HF model ID to local directory path."""
        local_name = model_id.replace("/", "--")
        return self.cache_dir / local_name

    def is_cached(self, model_id: str) -> bool:
        """Check if a model is already cached locally."""
        local_path = self.get_local_path(model_id)
        if not local_path.exists():
            return False

        has_model_files = (
            list(local_path.glob("*.bin"))
            or list(local_path.glob("*.safetensors"))
            or list(local_path.glob("pytorch_model.bin"))
            or list(local_path.glob("model.safetensors"))
        )
        return bool(has_model_files)

    def load_or_download(
        self, model_id: str, force_download: bool = False
    ) -> Path:
        """Load model from cache or download from HuggingFace Hub."""
        local_path = self.get_local_path(model_id)

        if not force_download and self.is_cached(model_id):
            logger.info("Loaded from cache: %s", local_path.name)
            return local_path

        logger.info("Downloading %s from HuggingFace Hub …", model_id)

        try:
            hf_model_path = snapshot_download(
                model_id,
                token=self.hf_token,
                cache_dir=None,
            )

            if hf_model_path != str(local_path):
                if local_path.exists():
                    shutil.rmtree(local_path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(hf_model_path, str(local_path))

            logger.info("Downloaded and cached: %s", local_path.name)
            return local_path

        except Exception as exc:
            logger.error("Error downloading %s: %s", model_id, exc)
            raise

    def clear_cache(self, model_id: Optional[str] = None) -> None:
        """Clear cached models (specific or all)."""
        if model_id:
            local_path = self.get_local_path(model_id)
            if local_path.exists():
                shutil.rmtree(local_path)
                logger.info("Cleared cache for %s", model_id)
            else:
                logger.warning("Model %s not found in cache", model_id)
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all cached models")

    def list_cached_models(self) -> List[str]:
        """List all cached models."""
        if not self.cache_dir.exists():
            return []

        cached_models: List[str] = []
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                model_id = item.name.replace("--", "/")
                cached_models.append(model_id)
        return cached_models

    def get_cache_size(self) -> int:
        """Get total size of cached models in bytes."""
        total_size = 0
        if self.cache_dir.exists():
            for item in self.cache_dir.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
        return total_size

    def get_cache_info(self) -> dict:
        """Get information about the cache."""
        cached_models = self.list_cached_models()
        cache_size = self.get_cache_size()
        return {
            "cache_dir": str(self.cache_dir.absolute()),
            "num_models": len(cached_models),
            "models": cached_models,
            "total_size_bytes": cache_size,
            "total_size_mb": cache_size / (1024 * 1024),
            "total_size_gb": cache_size / (1024 * 1024 * 1024),
        }


def disable_hf_repo_templates() -> None:
    """Disable HuggingFace repository templates check."""
    try:
        import transformers.utils.hub as hub
        hub.list_repo_templates = lambda *args, **kwargs: []
    except Exception:
        pass
