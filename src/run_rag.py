"""
Script to run the RAG system on HotPotQA dataset.

This script provides a command-line interface to:
1. Build the vector store from HotPotQA data
2. Query the RAG system interactively
3. Run batch evaluations
4. Run RAGAS evaluations
"""

import os
import argparse
from rag_system import RAGSystem
from evaluation.ragas_evaluator import RAGASEvaluator
from utils.file_utils import save_json, load_json
from config import RAGConfig


def build_vectorstore(args):
    """Build and save the vector store."""
    print("Building vector store...")
    
    rag = RAGSystem(
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k
    )
    
    # Load data - prioritize local file if it exists
    local_file = args.local_file or RAGConfig.DEFAULT_LOCAL_FILE
    if os.path.exists(local_file):
        print(f"Using local dataset: {local_file}")
        documents, questions_data = rag.load_data(
            local_file=local_file,
            max_samples=args.max_samples
        )
    else:
        print(f"Local file not found. Loading from HuggingFace ({args.split} split)...")
        documents, questions_data = rag.load_data(
            split=args.split,
            max_samples=args.max_samples
        )
    
    # Create and save vector store
    rag.create_vectorstore(documents, save_path=args.vectorstore_path)
    
    # Save questions data for later use
    questions_file = os.path.join(args.vectorstore_path, RAGConfig.DEFAULT_QUESTIONS_FILE)
    save_json(questions_data, questions_file)
    
    print(f"Vector store saved to: {args.vectorstore_path}")
    print(f"Questions data saved to: {questions_file}")


def interactive_query(args):
    """Run interactive query mode."""
    print("Loading RAG system...")
    
    rag = RAGSystem(
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k
    )
    
    # Load vector store
    rag.load_vectorstore(args.vectorstore_path)
    
    # Setup QA chain
    use_gemini = getattr(args, 'use_gemini', False)
    rag.setup_qa_chain(use_gemini=use_gemini)
    
    print("\n" + "="*80)
    print("Interactive RAG Query Mode")
    print("="*80)
    print("Type your questions below. Type 'quit' or 'exit' to stop.\n")
    
    while True:
        question = input("\nQuestion: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not question:
            continue
        
        try:
            result = rag.query(question)
            print(f"\nAnswer: {result['answer']}")
            
            # Check if answer contains an error
            if result['answer'].startswith("Error:"):
                print("\n⚠️  The LLM returned an error. Full error message:")
                print(result['answer'])
            
            print(f"\nRetrieved {len(result['source_documents'])} documents:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"\n  Document {i}:")
                print(f"    Title: {doc['metadata'].get('title', 'N/A')}")
                print(f"    Content: {doc['content'][:200]}...")
        except Exception as e:
            print(f"\n❌ Error during query: {e}")
            import traceback
            traceback.print_exc()


def batch_evaluation(args):
    """Run batch evaluation on test questions."""
    print("Loading RAG system...")
    
    rag = RAGSystem(
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k
    )
    
    # Load vector store
    rag.load_vectorstore(args.vectorstore_path)
    
    # Setup QA chain
    use_gemini = getattr(args, 'use_gemini', False)
    rag.setup_qa_chain(use_gemini=use_gemini)
    
    # Load questions
    questions_file = os.path.join(args.vectorstore_path, RAGConfig.DEFAULT_QUESTIONS_FILE)
    if not os.path.exists(questions_file):
        print(f"Questions file not found: {questions_file}")
        print("Please run 'build' mode first to create the vector store.")
        return
    
    questions_data = load_json(questions_file)
    
    # Limit number of questions if specified
    if args.num_questions:
        questions_data = questions_data[:args.num_questions]
    
    print(f"\nEvaluating {len(questions_data)} questions...")
    
    results = []
    for q_data in questions_data:
        result = rag.evaluate_sample(q_data['question'], q_data['answer'])
        results.append(result)
        
        # Print progress
        if len(results) % 10 == 0:
            print(f"Processed {len(results)}/{len(questions_data)} questions")
    
    # Save results
    output_file = args.output_file or "evaluation_results.json"
    save_json(results, output_file)
    
    # Print summary
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    print(f"Total questions: {len(results)}")
    print(f"Average retrieved documents: {sum(r['num_retrieved_docs'] for r in results) / len(results):.2f}")


def ragas_evaluation(args):
    """Run RAGAS evaluation on test questions."""
    print("Loading RAG system...")
    
    rag = RAGSystem(
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k
    )
    
    # Load vector store
    rag.load_vectorstore(args.vectorstore_path)
    
    # Setup QA chain
    use_gemini = getattr(args, 'use_gemini', False)
    rag.setup_qa_chain(use_gemini=use_gemini)
    
    # Load questions
    questions_file = os.path.join(args.vectorstore_path, RAGConfig.DEFAULT_QUESTIONS_FILE)
    if not os.path.exists(questions_file):
        print(f"Questions file not found: {questions_file}")
        print("Please run 'build' mode first to create the vector store.")
        return
    
    questions_data = load_json(questions_file)
    
    # Select metrics based on user input
    selected_metrics = None
    if args.ragas_metrics:
        selected_metrics = RAGASEvaluator.get_metrics_by_names(args.ragas_metrics)
    
    # Run RAGAS evaluation
    use_openai = getattr(args, 'use_openai_for_ragas', False)
    ragas_results = rag.evaluate_with_ragas(
        questions_data,
        metrics=selected_metrics,
        max_samples=args.num_questions,
        use_openai=use_openai
    )
    
    # Save results
    output_file = args.output_file or "ragas_evaluation_results.json"
    save_json(ragas_results, output_file)
    
    # Print summary
    print("\n" + "="*80)
    print("RAGAS Evaluation Summary")
    print("="*80)
    print(f"Total samples: {len(ragas_results.get('per_sample_scores', []))}")
    print("\nOverall Scores:")
    for metric_name, score in ragas_results['overall_scores'].items():
        print(f"  {metric_name}: {score:.4f}")
    
    # Print score ranges
    if ragas_results.get('per_sample_scores'):
        print("\nScore Ranges:")
        per_sample = ragas_results['per_sample_scores']
        for metric_name in ragas_results['overall_scores'].keys():
            scores = [s[metric_name] for s in per_sample if metric_name in s and s[metric_name] is not None]
            if scores:
                print(f"  {metric_name}: min={min(scores):.4f}, max={max(scores):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG System on HotPotQA")
    
    # Common arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["build", "query", "evaluate", "ragas"],
        required=True,
        help="Mode to run: build (create vectorstore), query (interactive), evaluate (batch), ragas (RAGAS evaluation)"
    )
    parser.add_argument(
        "--vectorstore-path",
        type=str,
        default=RAGConfig.DEFAULT_VECTORSTORE_PATH,
        help="Path to save/load vector store"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=RAGConfig.DEFAULT_EMBEDDING_MODEL,
        help="HuggingFace embedding model"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=RAGConfig.DEFAULT_LLM_MODEL,
        help="HuggingFace LLM model"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RAGConfig.DEFAULT_TOP_K,
        help="Number of documents to retrieve"
    )
    
    # Build mode arguments
    parser.add_argument(
        "--local-file",
        type=str,
        default=None,
        help=f"Path to local HotPotQA JSON file (default: {RAGConfig.DEFAULT_LOCAL_FILE})"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to use (only if local file not found)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load (None for all)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=RAGConfig.DEFAULT_CHUNK_SIZE,
        help="Size of text chunks"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=RAGConfig.DEFAULT_CHUNK_OVERLAP,
        help="Overlap between chunks"
    )
    
    # Evaluate mode arguments
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to evaluate (None for all)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for evaluation results (default: evaluation_results.json or ragas_evaluation_results.json)"
    )
    
    # RAGAS mode arguments
    parser.add_argument(
        "--ragas-metrics",
        type=str,
        nargs='+',
        choices=['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision', 'answer_correctness', 'answer_similarity'],
        default=None,
        help="RAGAS metrics to use (default: all metrics)"
    )
    parser.add_argument(
        "--use-openai-for-ragas",
        action='store_true',
        help="Use OpenAI for RAGAS evaluation (default: uses Google Gemini)"
    )
    parser.add_argument(
        "--use-gemini",
        action='store_true',
        help="Use Google Gemini for RAG generation instead of HuggingFace (recommended)"
    )
    
    args = parser.parse_args()
    
    # Check for HuggingFace API token if not using Gemini
    if args.mode in ["query", "evaluate", "ragas"] and not getattr(args, 'use_gemini', False):
        if not os.environ.get("HUGGINGFACEHUB_API_TOKEN") and not os.environ.get("HF_TOKEN"):
            print("WARNING: HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
            print("Please set it using: export HUGGINGFACEHUB_API_TOKEN=your_token")
            print("Or use --use-gemini flag to use Google Gemini instead.")
            return
    
    # Run appropriate mode
    if args.mode == "build":
        build_vectorstore(args)
    elif args.mode == "query":
        interactive_query(args)
    elif args.mode == "evaluate":
        batch_evaluation(args)
    elif args.mode == "ragas":
        ragas_evaluation(args)


if __name__ == "__main__":
    main()
