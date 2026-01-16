"""
Configuration settings for the RAG system.

This module contains all configuration constants and default values.
"""

import os
from typing import Dict, Any


class RAGConfig:
    """Configuration for RAG system."""
    
    # Model configurations
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"
    DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
    
    # Chunking configurations
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    
    # Retrieval configurations
    DEFAULT_TOP_K = 3
    DEFAULT_SEARCH_TYPE = "similarity"
    
    # LLM generation configurations
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_NEW_TOKENS = 512
    
    # Dataset configurations
    DEFAULT_DATASET_NAME = "hotpot_qa"
    DEFAULT_DATASET_CONFIG = "distractor"
    DEFAULT_LOCAL_FILE = "data/hotpot_dev_distractor_v1.json"
    
    # Storage paths
    DEFAULT_VECTORSTORE_PATH = "./vectorstore"
    DEFAULT_QUESTIONS_FILE = "questions.json"
    
    # API configurations
    HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/{model}"
    HUGGINGFACE_API_TIMEOUT = 120
    
    @staticmethod
    def get_api_keys() -> Dict[str, str]:
        """Get API keys from environment variables."""
        return {
            "huggingface": os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN"),
            "google": os.getenv("GOOGLE_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY")
        }
    
    @staticmethod
    def validate_api_key(provider: str) -> bool:
        """Validate that required API key exists."""
        keys = RAGConfig.get_api_keys()
        return keys.get(provider) is not None


class PromptTemplates:
    """Prompt templates for the RAG system."""
    
    QA_TEMPLATE = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
    
    @staticmethod
    def get_qa_template() -> str:
        """Get the QA prompt template."""
        return PromptTemplates.QA_TEMPLATE
