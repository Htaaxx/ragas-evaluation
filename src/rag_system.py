"""
Core RAG (Retrieval-Augmented Generation) System.

This module provides the main RAG system class that orchestrates
data loading, vector storage, retrieval, and generation.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from dotenv import load_dotenv

from .config import RAGConfig, PromptTemplates
from .data.loader import DataLoader
from .vectorstore.manager import VectorStoreManager
from .models.huggingface_llm import HuggingFaceInferenceLLM
from .models.gemini_llm import create_gemini_llm
from .evaluation.ragas_evaluator import RAGASEvaluator
from .utils.file_utils import save_json, load_json

# Load environment variables
load_dotenv()


class RAGSystem:
    """
    A RAG system for question answering on HotPotQA dataset.
    
    This class orchestrates the entire RAG pipeline including:
    - Data loading
    - Vector store creation and management
    - Question answering with retrieval
    - Evaluation using RAGAS
    """
    
    def __init__(
        self,
        embedding_model: str = RAGConfig.DEFAULT_EMBEDDING_MODEL,
        llm_model: str = RAGConfig.DEFAULT_LLM_MODEL,
        chunk_size: int = RAGConfig.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = RAGConfig.DEFAULT_CHUNK_OVERLAP,
        top_k: int = RAGConfig.DEFAULT_TOP_K
    ):
        """Initialize the RAG system.
        
        Args:
            embedding_model: Name of the HuggingFace embedding model
            llm_model: Name of the HuggingFace language model
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
        """
        self.llm_model_name = llm_model
        self.top_k = top_k
        
        # Initialize vector store manager
        self.vectorstore_manager = VectorStoreManager(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize components
        self.qa_chain = None
        self.evaluator = None
    
    def load_data(
        self,
        split: str = "train",
        max_samples: Optional[int] = 1000,
        local_file: Optional[str] = None
    ):
        """Load HotPotQA dataset.
        
        Args:
            split: Dataset split to load ('train', 'validation')
            max_samples: Maximum number of samples to load (None for all)
            local_file: Path to local JSON file (if None, loads from HuggingFace)
            
        Returns:
            Tuple of (documents, questions_data)
        """
        return DataLoader.load_hotpotqa(
            split=split,
            max_samples=max_samples,
            local_file=local_file
        )
    
    def create_vectorstore(self, documents: List, save_path: Optional[str] = None):
        """Create a vector store from documents.
        
        Args:
            documents: List of Document objects
            save_path: Optional path to save the vectorstore
        """
        self.vectorstore_manager.create_vectorstore(documents, save_path)
    
    def load_vectorstore(self, load_path: str):
        """Load a pre-existing vector store.
        
        Args:
            load_path: Path to the saved vectorstore
        """
        self.vectorstore_manager.load_vectorstore(load_path)
    
    def setup_qa_chain(
        self,
        huggingface_api_token: Optional[str] = None,
        use_gemini: bool = False
    ):
        """Set up the QA chain with retriever and LLM.
        
        Args:
            huggingface_api_token: HuggingFace API token (optional, uses env var if not provided)
            use_gemini: If True, uses Google Gemini instead of HuggingFace (recommended)
        """
        if self.vectorstore_manager.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_vectorstore first.")
        
        print("Setting up QA chain...")
        
        # Set up retriever
        retriever = self.vectorstore_manager.get_retriever(top_k=self.top_k)
        
        # Set up LLM
        if use_gemini:
            # Use Google Gemini (more reliable than HuggingFace free tier)
            print("Using Google Gemini for generation...")
            llm = create_gemini_llm(temperature=RAGConfig.DEFAULT_TEMPERATURE)
        else:
            # Use HuggingFace Inference API
            if huggingface_api_token:
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token
            elif not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
                # Try alternative environment variable name
                token = os.getenv("HF_TOKEN")
                if token:
                    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
                else:
                    raise ValueError(
                        "HuggingFace API token not found. Please either:\n"
                        "1. Set HUGGINGFACEHUB_API_TOKEN in your .env file\n"
                        "2. Pass huggingface_api_token parameter\n"
                        "3. Set HF_TOKEN environment variable\n"
                        "4. Use --use-gemini flag to use Google Gemini instead"
                    )
            
            # Use custom HuggingFace Inference LLM with direct API calls
            llm = HuggingFaceInferenceLLM(
                model=self.llm_model_name,
                temperature=RAGConfig.DEFAULT_TEMPERATURE,
                max_new_tokens=RAGConfig.DEFAULT_MAX_NEW_TOKENS,
                token=os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
            )
        
        # Create custom prompt
        PROMPT = PromptTemplate(
            template=PromptTemplates.get_qa_template(),
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("QA chain setup complete!")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a question.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary containing answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call setup_qa_chain first.")
        
        # Use invoke instead of __call__ (deprecated in langchain 0.1.0)
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "question": question,
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        }
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Query the RAG system with multiple questions.
        
        Args:
            questions: List of questions to answer
            
        Returns:
            List of results for each question
        """
        results = []
        for question in tqdm(questions, desc="Processing questions"):
            try:
                result = self.query(question)
                results.append(result)
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                results.append({
                    "question": question,
                    "answer": "Error",
                    "error": str(e)
                })
        
        return results
    
    def evaluate_sample(self, question: str, ground_truth: str) -> Dict[str, Any]:
        """Evaluate a single question against ground truth.
        
        Args:
            question: The question
            ground_truth: The correct answer
            
        Returns:
            Dictionary with question, predicted answer, ground truth, and retrieved docs
        """
        result = self.query(question)
        
        return {
            "question": question,
            "predicted_answer": result["answer"],
            "ground_truth": ground_truth,
            "source_documents": result["source_documents"],
            "num_retrieved_docs": len(result["source_documents"])
        }
    
    def evaluate_with_ragas(
        self,
        questions_data: List[Dict[str, Any]],
        metrics: Optional[List[Any]] = None,
        max_samples: Optional[int] = None,
        use_openai: bool = False
    ) -> Dict[str, Any]:
        """Evaluate RAG system using RAGAS framework.
        
        Args:
            questions_data: List of dicts with 'question' and 'answer' keys
            metrics: List of RAGAS metrics to use. If None, uses all available metrics.
            max_samples: Maximum number of samples to evaluate (None for all)
            use_openai: If True, uses OpenAI for evaluation. If False, uses Google Gemini (default).
            
        Returns:
            Dictionary containing RAGAS evaluation results and per-sample scores
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call setup_qa_chain first.")
        
        # Initialize evaluator if not already done
        if self.evaluator is None:
            self.evaluator = RAGASEvaluator(self.vectorstore_manager.embeddings)
        
        # Limit samples if specified
        if max_samples:
            questions_data = questions_data[:max_samples]
        
        print(f"\nEvaluating {len(questions_data)} samples with RAGAS...")
        
        # Prepare data for RAGAS
        ragas_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        # Query the RAG system for each question
        for idx, q_data in enumerate(tqdm(questions_data, desc="Querying RAG system")):
            try:
                result = self.query(q_data['question'])
                
                # Extract contexts from source documents
                contexts = [doc['content'] for doc in result['source_documents']]
                
                ragas_data["question"].append(q_data['question'])
                ragas_data["answer"].append(result['answer'])
                ragas_data["contexts"].append(contexts)
                ragas_data["ground_truth"].append(q_data['answer'])
                
            except Exception as e:
                print(f"\nError processing question {idx}: {e}")
                # Add placeholder data to maintain alignment
                ragas_data["question"].append(q_data['question'])
                ragas_data["answer"].append("Error: Could not generate answer")
                ragas_data["contexts"].append(["Error: Could not retrieve context"])
                ragas_data["ground_truth"].append(q_data['answer'])
        
        # Run RAGAS evaluation
        return self.evaluator.evaluate(ragas_data, metrics=metrics, use_openai=use_openai)


def main():
    """Main function - Use run_rag.py instead for better control."""
    import sys
    print("=" * 80)
    print("NOTE: Please use run_rag.py for running the RAG system")
    print("=" * 80)
    print("\nExamples:")
    print("  Build vectorstore:  python src/run_rag.py --mode build")
    print("  Interactive query:  python src/run_rag.py --mode query")
    print("  Batch evaluation:   python src/run_rag.py --mode evaluate")
    print("\nFor more options: python src/run_rag.py --help")
    sys.exit(0)


if __name__ == "__main__":
    main()
