"""
Main Entry Point for RAG System

This is the primary interface for running the RAG system.
It provides an interactive menu for selecting modes and configuring settings.

Usage:
    python main.py                    # Interactive mode
    python main.py --help             # Show all options
    python main.py --mode build       # Direct mode selection
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_system import RAGSystem
from src.evaluation.ragas_evaluator import RAGASEvaluator
from src.utils.file_utils import save_json, load_json
from src.config import RAGConfig


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*80)
    print("🤖 RAG System for HotPotQA - Interactive Interface")
    print("="*80 + "\n")


def print_menu():
    """Print main menu."""
    print("\n📋 Available Modes:")
    print("  1. Build Vector Store    - Create searchable index from documents")
    print("  2. Interactive Query     - Ask questions interactively")
    print("  3. Batch Evaluation      - Evaluate multiple questions")
    print("  4. RAGAS Evaluation      - Comprehensive RAG metrics")
    print("  5. Configuration Info    - View current settings")
    print("  6. Exit")
    print()


def get_user_choice(prompt, choices, default=None):
    """Get user choice with validation."""
    while True:
        if default:
            user_input = input(f"{prompt} (default: {default}): ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        if user_input in choices or user_input in [str(i) for i in range(1, len(choices) + 1)]:
            return user_input
        print(f"❌ Invalid choice. Please choose from: {', '.join(choices)}")


def get_number_input(prompt, default=None, min_val=None, max_val=None):
    """Get numeric input with validation."""
    while True:
        if default is not None:
            user_input = input(f"{prompt} (default: {default}): ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        try:
            value = int(user_input)
            if min_val is not None and value < min_val:
                print(f"❌ Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"❌ Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("❌ Please enter a valid number")


def get_yes_no(prompt, default="n"):
    """Get yes/no input."""
    while True:
        user_input = input(f"{prompt} (y/n, default: {default}): ").strip().lower()
        if not user_input:
            return default == "y"
        if user_input in ["y", "yes"]:
            return True
        if user_input in ["n", "no"]:
            return False
        print("❌ Please enter 'y' or 'n'")


def mode_build():
    """Build vector store mode."""
    print("\n" + "="*80)
    print("🏗️  BUILD VECTOR STORE")
    print("="*80 + "\n")
    
    print("This will create a searchable index from the HotPotQA dataset.")
    print("The dataset contains 7,405 questions with supporting documents.\n")
    
    # Get configuration
    max_samples = get_number_input(
        "How many samples to process? (Enter 0 for all)",
        default=1000,
        min_val=0
    )
    if max_samples == 0:
        max_samples = None
        print("⚠️  Processing all 7,405 samples will take 10-15 minutes")
    
    chunk_size = get_number_input(
        "Chunk size for text splitting",
        default=RAGConfig.DEFAULT_CHUNK_SIZE,
        min_val=100,
        max_val=2000
    )
    
    chunk_overlap = get_number_input(
        "Chunk overlap",
        default=RAGConfig.DEFAULT_CHUNK_OVERLAP,
        min_val=0,
        max_val=chunk_size // 2
    )
    
    vectorstore_path = input(
        f"Vector store save path (default: {RAGConfig.DEFAULT_VECTORSTORE_PATH}): "
    ).strip() or RAGConfig.DEFAULT_VECTORSTORE_PATH
    
    # Confirm
    print("\n📋 Configuration Summary:")
    print(f"  • Samples: {max_samples if max_samples else 'All (7,405)'}")
    print(f"  • Chunk size: {chunk_size}")
    print(f"  • Chunk overlap: {chunk_overlap}")
    print(f"  • Save path: {vectorstore_path}")
    
    if not get_yes_no("\nProceed with build?", default="y"):
        print("❌ Build cancelled")
        return
    
    # Build
    print("\n🚀 Starting build...\n")
    
    try:
        rag = RAGSystem(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Load data
        local_file = RAGConfig.DEFAULT_LOCAL_FILE
        if os.path.exists(local_file):
            print(f"📂 Using local dataset: {local_file}")
            documents, questions_data = rag.load_data(
                local_file=local_file,
                max_samples=max_samples
            )
        else:
            print("📥 Loading from HuggingFace...")
            documents, questions_data = rag.load_data(
                split="train",
                max_samples=max_samples
            )
        
        # Create vector store
        rag.create_vectorstore(documents, save_path=vectorstore_path)
        
        # Save questions
        questions_file = os.path.join(vectorstore_path, RAGConfig.DEFAULT_QUESTIONS_FILE)
        save_json(questions_data, questions_file)
        
        print("\n✅ Build complete!")
        print(f"📁 Vector store saved to: {vectorstore_path}")
        print(f"📄 Questions saved to: {questions_file}")
        
    except Exception as e:
        print(f"\n❌ Error during build: {e}")
        import traceback
        traceback.print_exc()


def mode_query():
    """Interactive query mode."""
    print("\n" + "="*80)
    print("💬 INTERACTIVE QUERY")
    print("="*80 + "\n")
    
    # Check vector store
    vectorstore_path = RAGConfig.DEFAULT_VECTORSTORE_PATH
    if not os.path.exists(vectorstore_path):
        print(f"❌ Vector store not found at: {vectorstore_path}")
        print("Please run 'Build Vector Store' first.")
        return
    
    # Get configuration
    print("Choose LLM provider:")
    print("  1. Google Gemini (recommended - more reliable)")
    print("  2. HuggingFace Inference API")
    
    llm_choice = get_user_choice("Select provider", ["1", "2"], default="1")
    use_gemini = llm_choice == "1"
    
    if use_gemini:
        if not os.getenv("GOOGLE_API_KEY"):
            print("\n⚠️  GOOGLE_API_KEY not found!")
            print("Please set it: $env:GOOGLE_API_KEY='your-key'")
            print("Get your key at: https://aistudio.google.com/apikey")
            return
        print("✓ Using Google Gemini")
    else:
        if not os.getenv("HUGGINGFACEHUB_API_TOKEN") and not os.getenv("HF_TOKEN"):
            print("\n⚠️  HUGGINGFACEHUB_API_TOKEN not found!")
            print("Please set it: $env:HUGGINGFACEHUB_API_TOKEN='your-token'")
            return
        print("✓ Using HuggingFace")
    
    top_k = get_number_input(
        "Number of documents to retrieve",
        default=RAGConfig.DEFAULT_TOP_K,
        min_val=1,
        max_val=10
    )
    
    # Load RAG system
    print("\n🚀 Loading RAG system...")
    
    try:
        rag = RAGSystem(top_k=top_k)
        rag.load_vectorstore(vectorstore_path)
        rag.setup_qa_chain(use_gemini=use_gemini)
        
        print("\n✅ System ready!")
        print("\n" + "="*80)
        print("💬 Ask your questions below. Type 'quit', 'exit', or 'q' to stop.")
        print("="*80 + "\n")
        
        # Query loop
        while True:
            question = input("❓ Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            try:
                result = rag.query(question)
                
                print(f"\n💡 Answer: {result['answer']}")
                
                if result['answer'].startswith("Error:"):
                    print("\n⚠️  The LLM returned an error.")
                
                print(f"\n📚 Retrieved {len(result['source_documents'])} documents:")
                for i, doc in enumerate(result['source_documents'], 1):
                    print(f"\n  📄 Document {i}:")
                    print(f"     Title: {doc['metadata'].get('title', 'N/A')}")
                    content_preview = doc['content'][:150] + "..." if len(doc['content']) > 150 else doc['content']
                    print(f"     Content: {content_preview}")
                
                print()
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
    except Exception as e:
        print(f"\n❌ Error loading system: {e}")
        import traceback
        traceback.print_exc()


def mode_evaluate():
    """Batch evaluation mode."""
    print("\n" + "="*80)
    print("📊 BATCH EVALUATION")
    print("="*80 + "\n")
    
    # Check vector store
    vectorstore_path = RAGConfig.DEFAULT_VECTORSTORE_PATH
    if not os.path.exists(vectorstore_path):
        print(f"❌ Vector store not found at: {vectorstore_path}")
        print("Please run 'Build Vector Store' first.")
        return
    
    # Check questions file
    questions_file = os.path.join(vectorstore_path, RAGConfig.DEFAULT_QUESTIONS_FILE)
    if not os.path.exists(questions_file):
        print(f"❌ Questions file not found: {questions_file}")
        print("Please run 'Build Vector Store' first.")
        return
    
    # Get configuration
    print("Choose LLM provider:")
    print("  1. Google Gemini (recommended)")
    print("  2. HuggingFace Inference API")
    
    llm_choice = get_user_choice("Select provider", ["1", "2"], default="1")
    use_gemini = llm_choice == "1"
    
    if use_gemini and not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠️  GOOGLE_API_KEY not found!")
        return
    
    num_questions = get_number_input(
        "Number of questions to evaluate (0 for all)",
        default=10,
        min_val=0
    )
    if num_questions == 0:
        num_questions = None
    
    output_file = input(
        "Output file (default: evaluation_results.json): "
    ).strip() or "evaluation_results.json"
    
    # Confirm
    print("\n📋 Configuration Summary:")
    print(f"  • LLM: {'Google Gemini' if use_gemini else 'HuggingFace'}")
    print(f"  • Questions: {num_questions if num_questions else 'All'}")
    print(f"  • Output: {output_file}")
    
    if not get_yes_no("\nProceed with evaluation?", default="y"):
        print("❌ Evaluation cancelled")
        return
    
    # Evaluate
    print("\n🚀 Starting evaluation...\n")
    
    try:
        rag = RAGSystem()
        rag.load_vectorstore(vectorstore_path)
        rag.setup_qa_chain(use_gemini=use_gemini)
        
        questions_data = load_json(questions_file)
        if num_questions:
            questions_data = questions_data[:num_questions]
        
        print(f"Evaluating {len(questions_data)} questions...")
        
        results = []
        for i, q_data in enumerate(questions_data, 1):
            result = rag.evaluate_sample(q_data['question'], q_data['answer'])
            results.append(result)
            if i % 10 == 0:
                print(f"  Processed {i}/{len(questions_data)} questions")
        
        save_json(results, output_file)
        
        print("\n✅ Evaluation complete!")
        print(f"📁 Results saved to: {output_file}")
        print(f"\n📊 Summary:")
        print(f"  • Total questions: {len(results)}")
        avg_docs = sum(r['num_retrieved_docs'] for r in results) / len(results)
        print(f"  • Avg retrieved docs: {avg_docs:.2f}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


def mode_ragas():
    """RAGAS evaluation mode."""
    print("\n" + "="*80)
    print("📈 RAGAS EVALUATION")
    print("="*80 + "\n")
    
    print("RAGAS provides comprehensive metrics for RAG systems:")
    print("  • Faithfulness - Answer grounded in context")
    print("  • Answer Relevancy - Answer relevant to question")
    print("  • Context Precision - Retrieved docs are relevant")
    print("  • Context Recall - All relevant info retrieved")
    print("  • Answer Correctness - Quality vs ground truth")
    print("  • Answer Similarity - Semantic similarity\n")
    
    # Check vector store
    vectorstore_path = RAGConfig.DEFAULT_VECTORSTORE_PATH
    if not os.path.exists(vectorstore_path):
        print(f"❌ Vector store not found at: {vectorstore_path}")
        print("Please run 'Build Vector Store' first.")
        return
    
    # Check questions file
    questions_file = os.path.join(vectorstore_path, RAGConfig.DEFAULT_QUESTIONS_FILE)
    if not os.path.exists(questions_file):
        print(f"❌ Questions file not found: {questions_file}")
        return
    
    # Get configuration
    print("Choose LLM provider for RAG generation:")
    print("  1. Google Gemini (recommended)")
    print("  2. HuggingFace Inference API")
    
    llm_choice = get_user_choice("Select provider", ["1", "2"], default="1")
    use_gemini = llm_choice == "1"
    
    if use_gemini and not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠️  GOOGLE_API_KEY not found!")
        print("Get your key at: https://aistudio.google.com/apikey")
        return
    
    num_questions = get_number_input(
        "Number of questions to evaluate",
        default=5,
        min_val=1,
        max_val=100
    )
    
    use_all_metrics = get_yes_no("Use all RAGAS metrics?", default="y")
    
    selected_metrics = None
    if not use_all_metrics:
        print("\nAvailable metrics:")
        print("  1. faithfulness")
        print("  2. answer_relevancy")
        print("  3. context_recall")
        print("  4. context_precision")
        print("  5. answer_correctness")
        print("  6. answer_similarity")
        metric_input = input("Enter metric numbers (comma-separated, e.g., 1,2,3): ").strip()
        
        metric_map = {
            "1": "faithfulness",
            "2": "answer_relevancy",
            "3": "context_recall",
            "4": "context_precision",
            "5": "answer_correctness",
            "6": "answer_similarity"
        }
        
        selected_metrics = [metric_map[m.strip()] for m in metric_input.split(",") if m.strip() in metric_map]
    
    output_file = input(
        "Output file (default: ragas_evaluation_results.json): "
    ).strip() or "ragas_evaluation_results.json"
    
    # Confirm
    print("\n📋 Configuration Summary:")
    print(f"  • LLM: {'Google Gemini' if use_gemini else 'HuggingFace'}")
    print(f"  • Questions: {num_questions}")
    print(f"  • Metrics: {'All' if use_all_metrics else ', '.join(selected_metrics)}")
    print(f"  • Output: {output_file}")
    
    if not get_yes_no("\nProceed with RAGAS evaluation?", default="y"):
        print("❌ Evaluation cancelled")
        return
    
    # Evaluate
    print("\n🚀 Starting RAGAS evaluation...\n")
    
    try:
        rag = RAGSystem()
        rag.load_vectorstore(vectorstore_path)
        rag.setup_qa_chain(use_gemini=use_gemini)
        
        questions_data = load_json(questions_file)
        
        # Get metrics
        metrics = None
        if selected_metrics:
            metrics = RAGASEvaluator.get_metrics_by_names(selected_metrics)
        
        ragas_results = rag.evaluate_with_ragas(
            questions_data,
            metrics=metrics,
            max_samples=num_questions,
            use_openai=False
        )
        
        save_json(ragas_results, output_file)
        
        print("\n✅ RAGAS evaluation complete!")
        print(f"📁 Results saved to: {output_file}")
        print(f"\n📊 Overall Scores:")
        for metric_name, score in ragas_results['overall_scores'].items():
            print(f"  • {metric_name}: {score:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error during RAGAS evaluation: {e}")
        import traceback
        traceback.print_exc()


def show_config():
    """Show current configuration."""
    print("\n" + "="*80)
    print("⚙️  CURRENT CONFIGURATION")
    print("="*80 + "\n")
    
    print("📋 Default Settings:")
    print(f"  • Embedding Model: {RAGConfig.DEFAULT_EMBEDDING_MODEL}")
    print(f"  • LLM Model: {RAGConfig.DEFAULT_LLM_MODEL}")
    print(f"  • Gemini Model: {RAGConfig.DEFAULT_GEMINI_MODEL}")
    print(f"  • Chunk Size: {RAGConfig.DEFAULT_CHUNK_SIZE}")
    print(f"  • Chunk Overlap: {RAGConfig.DEFAULT_CHUNK_OVERLAP}")
    print(f"  • Top K: {RAGConfig.DEFAULT_TOP_K}")
    print(f"  • Temperature: {RAGConfig.DEFAULT_TEMPERATURE}")
    print(f"  • Max New Tokens: {RAGConfig.DEFAULT_MAX_NEW_TOKENS}")
    
    print("\n🔑 API Keys Status:")
    api_keys = RAGConfig.get_api_keys()
    print(f"  • HuggingFace: {'✓ Set' if api_keys['huggingface'] else '✗ Not set'}")
    print(f"  • Google (Gemini): {'✓ Set' if api_keys['google'] else '✗ Not set'}")
    print(f"  • OpenAI: {'✓ Set' if api_keys['openai'] else '✗ Not set (optional)'}")
    
    print("\n📁 Paths:")
    print(f"  • Vector Store: {RAGConfig.DEFAULT_VECTORSTORE_PATH}")
    print(f"  • Local Dataset: {RAGConfig.DEFAULT_LOCAL_FILE}")
    
    print("\n💡 To modify configuration, edit: src/config.py")
    
    input("\nPress Enter to continue...")


def interactive_mode():
    """Run in interactive mode."""
    print_banner()
    
    while True:
        print_menu()
        choice = input("Select mode (1-6): ").strip()
        
        if choice == "1":
            mode_build()
        elif choice == "2":
            mode_query()
        elif choice == "3":
            mode_evaluate()
        elif choice == "4":
            mode_ragas()
        elif choice == "5":
            show_config()
        elif choice == "6":
            print("\n👋 Goodbye!\n")
            break
        else:
            print("❌ Invalid choice. Please select 1-6.")
        
        if choice in ["1", "2", "3", "4"]:
            input("\nPress Enter to return to main menu...")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RAG System for HotPotQA - Main Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive mode
  python main.py --mode build             # Build vector store
  python main.py --mode query             # Interactive query
  python main.py --mode evaluate          # Batch evaluation
  python main.py --mode ragas             # RAGAS evaluation
  python main.py --config                 # Show configuration

For more control, use: python src/run_rag.py --help
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["build", "query", "evaluate", "ragas"],
        help="Run mode directly (skip interactive menu)"
    )
    
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show configuration and exit"
    )
    
    args = parser.parse_args()
    
    # Handle direct mode selection
    if args.config:
        print_banner()
        show_config()
        return
    
    if args.mode:
        print_banner()
        if args.mode == "build":
            mode_build()
        elif args.mode == "query":
            mode_query()
        elif args.mode == "evaluate":
            mode_evaluate()
        elif args.mode == "ragas":
            mode_ragas()
        return
    
    # Interactive mode
    interactive_mode()


if __name__ == "__main__":
    main()
