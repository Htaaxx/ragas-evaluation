"""
File utility functions.

This module provides utilities for file operations.
"""

import json
import os
from typing import Any, Dict, List


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save the file
        indent: JSON indentation level
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    print(f"Data saved to: {filepath}")


def load_json(filepath: str) -> Any:
    """Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
