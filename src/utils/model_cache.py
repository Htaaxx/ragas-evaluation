"""
Model caching utilities for automatic model download and local storage.

This module handles downloading models from HuggingFace Hub and caching them
locally to avoid repeated downloads.
"""

import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


class ModelCache:
    """
    Manager for caching HuggingFace models locally.
    
    Features:
    - Automatic download from HuggingFace Hub
    - Local caching to avoid repeated downloads
    - Organized storage with model ID as directory name
    """
    
    def __init__(self, cache_dir: Path, hf_token: Optional[str] = None):
        """
        Initialize the model cache manager.
        
        Args:
            cache_dir: Directory to store cached models
            hf_token: HuggingFace API token (optional, for private models)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token
        
        print(f"Model cache directory: {self.cache_dir.absolute()}")
    
    def get_local_path(self, model_id: str) -> Path:
        """
        Get local path for a model.
        
        Converts HuggingFace model ID to local directory name.
        Example: "sentence-transformers/all-MiniLM-L6-v2" 
                 -> "sentence-transformers--all-MiniLM-L6-v2"
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            Path to local model directory
        """
        # Replace slashes with double dashes for filesystem compatibility
        local_name = model_id.replace("/", "--")
        return self.cache_dir / local_name
    
    def is_cached(self, model_id: str) -> bool:
        """
        Check if a model is already cached locally.
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            True if model exists in cache, False otherwise
        """
        local_path = self.get_local_path(model_id)
        
        # Check if directory exists and contains model files
        if not local_path.exists():
            return False
        
        # Check for common model file extensions
        has_model_files = (
            list(local_path.glob("*.bin")) or 
            list(local_path.glob("*.safetensors")) or
            list(local_path.glob("pytorch_model.bin")) or
            list(local_path.glob("model.safetensors"))
        )
        
        return bool(has_model_files)
    
    def load_or_download(self, model_id: str, force_download: bool = False) -> Path:
        """
        Load model from cache or download from HuggingFace Hub.
        
        Args:
            model_id: HuggingFace model ID
            force_download: Force re-download even if cached
            
        Returns:
            Path to local model directory
            
        Raises:
            Exception: If download fails
        """
        local_path = self.get_local_path(model_id)
        
        # Check if already cached
        if not force_download and self.is_cached(model_id):
            print(f"Loaded from cache: {local_path.name}")
            return local_path
        
        # Download from HuggingFace Hub
        print(f"Downloading {model_id} from HuggingFace Hub...")
        
        try:
            # Download model to temporary location
            hf_model_path = snapshot_download(
                model_id,
                token=self.hf_token,
                cache_dir=None,  # Use HF default cache
            )
            
            # Move to our cache directory
            if hf_model_path != str(local_path):
                # Remove existing cache if present
                if local_path.exists():
                    shutil.rmtree(local_path)
                
                # Move downloaded model to cache
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(hf_model_path, str(local_path))
            
            print(f"Downloaded and cached: {local_path.name}")
            return local_path
            
        except Exception as e:
            print(f"Error downloading {model_id}: {e}")
            raise
    
    def clear_cache(self, model_id: Optional[str] = None) -> None:
        """
        Clear cached models.
        
        Args:
            model_id: Specific model to clear (None = clear all)
        """
        if model_id:
            # Clear specific model
            local_path = self.get_local_path(model_id)
            if local_path.exists():
                shutil.rmtree(local_path)
                print(f"Cleared cache for {model_id}")
            else:
                print(f"Warning: Model {model_id} not found in cache")
        else:
            # Clear all cached models
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                print("Cleared all cached models")
    
    def list_cached_models(self) -> list:
        """
        List all cached models.
        
        Returns:
            List of model IDs in cache
        """
        if not self.cache_dir.exists():
            return []
        
        cached_models = []
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                # Convert directory name back to model ID
                model_id = item.name.replace("--", "/")
                cached_models.append(model_id)
        
        return cached_models
    
    def get_cache_size(self) -> int:
        """
        Get total size of cached models in bytes.
        
        Returns:
            Total cache size in bytes
        """
        total_size = 0
        
        if self.cache_dir.exists():
            for item in self.cache_dir.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
        
        return total_size
    
    def get_cache_info(self) -> dict:
        """
        Get information about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
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


def disable_hf_repo_templates():
    """
    Disable HuggingFace repository templates check.
    
    This prevents unnecessary API calls during model loading.
    """
    try:
        import transformers.utils.hub as hub
        hub.list_repo_templates = lambda *args, **kwargs: []
    except Exception:
        pass  # Silently ignore if not available
