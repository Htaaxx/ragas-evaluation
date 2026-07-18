"""
Main training script for RAG system.

This script provides a complete training pipeline:
1. Load data
2. Train retriever
3. Evaluate retriever
4. Build index
5. Train generator (optional)
6. Create QA pipeline

Usage:
    python train.py --train-retriever --train-generator
    python train.py --train-retriever-only
    python train.py --config custom_config.json
"""

import argparse
import json
from pathlib import Path

from src.config import RAGConfig
from src.rag_system import RAGSystem


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RAG system")
    
    # Data paths
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/hotpot_qa/train.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--valid-data",
        type=str,
        default="data/hotpot_qa/valid.csv",
        help="Path to validation data CSV"
    )
    
    # Training options
    parser.add_argument(
        "--train-retriever",
        action="store_true",
        help="Train the retriever model"
    )
    parser.add_argument(
        "--train-generator",
        action="store_true",
        help="Train the generator model"
    )
    parser.add_argument(
        "--train-retriever-only",
        action="store_true",
        help="Train only the retriever (skip generator)"
    )
    
    # Model selection
    parser.add_argument(
        "--encoder-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Encoder model ID"
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="google/flan-t5-base",
        help="Generator model ID"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--retriever-epochs",
        type=int,
        default=5,
        help="Number of retriever training epochs"
    )
    parser.add_argument(
        "--retriever-batch-size",
        type=int,
        default=16,
        help="Retriever batch size"
    )
    parser.add_argument(
        "--retriever-lr",
        type=float,
        default=2e-5,
        help="Retriever learning rate"
    )
    parser.add_argument(
        "--generator-epochs",
        type=int,
        default=3,
        help="Number of generator training epochs"
    )
    parser.add_argument(
        "--generator-batch-size",
        type=int,
        default=4,
        help="Generator batch size"
    )
    parser.add_argument(
        "--generator-lr",
        type=float,
        default=5e-5,
        help="Generator learning rate"
    )
    parser.add_argument(
        "--generator-max-examples",
        type=int,
        default=None,
        help="Maximum generator training examples (None = all)"
    )
    
    # Paths
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./rag_output",
        help="Output directory"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="../models",
        help="Models cache directory"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file"
    )
    
    # Evaluation
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip retriever evaluation"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = RAGConfig.from_dict(config_dict)
    else:
        config = RAGConfig()
    
    # Override config with command line arguments
    config.encoder_model = args.encoder_model
    config.generator_model = args.generator_model
    config.train_data_path = args.train_data
    config.valid_data_path = args.valid_data
    config.output_dir = Path(args.output_dir)
    config.models_dir = Path(args.models_dir)
    
    if args.device:
        config.device = args.device
    
    # Update training hyperparameters
    config.retriever_epochs = args.retriever_epochs
    config.retriever_batch_size = args.retriever_batch_size
    config.retriever_lr = args.retriever_lr
    config.generator_epochs = args.generator_epochs
    config.generator_batch_size = args.generator_batch_size
    config.generator_lr = args.generator_lr
    
    # Initialize RAG system
    rag = RAGSystem(config=config)
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    rag.load_data()
    
    # Determine what to train
    train_retriever = args.train_retriever or args.train_retriever_only
    train_generator = args.train_generator and not args.train_retriever_only
    
    if not train_retriever and not train_generator:
        print("\nWarning: No training specified. Use --train-retriever and/or --train-generator")
        print("   Example: python train.py --train-retriever --train-generator")
        return
    
    # Train retriever
    if train_retriever:
        rag.train_retriever(
            epochs=args.retriever_epochs,
            batch_size=args.retriever_batch_size,
            lr=args.retriever_lr
        )
        
        # Evaluate retriever
        if not args.skip_eval:
            rag.evaluate_retriever()
    
    # Build index (required for generator training and QA)
    if train_generator or not args.skip_eval:
        rag.build_index()
    
    # Train generator
    if train_generator:
        rag.train_generator(
            epochs=args.generator_epochs,
            batch_size=args.generator_batch_size,
            lr=args.generator_lr,
            max_examples=args.generator_max_examples
        )
    
    # Test QA pipeline
    print("\n" + "="*60)
    print("Testing QA Pipeline")
    print("="*60)
    
    test_questions = [
        "What is the capital of France?",
        "Who founded Microsoft?",
        "Where is the Eiffel Tower located?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        try:
            answer, contexts = rag.answer(question, return_contexts=True)
            print(f"Answer: {answer}")
            print(f"Contexts used: {len(contexts)}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModels saved to: {config.models_dir}")
    print(f"Index saved to: {config.index_dir}")
    print(f"Outputs saved to: {config.output_dir}")
    
    # Save configuration
    config_save_path = config.output_dir / "config.json"
    with open(config_save_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Configuration saved to: {config_save_path}")


if __name__ == "__main__":
    main()
