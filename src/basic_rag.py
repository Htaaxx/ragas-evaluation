"""
DEPRECATED: This file is kept for backward compatibility only.

Please use the new modular structure:
- src/rag_system.py - Main RAG system class
- src/models/ - LLM wrappers
- src/data/ - Data loading
- src/vectorstore/ - Vector store management
- src/evaluation/ - RAGAS evaluation
- src/config.py - Configuration
- src/utils/ - Utility functions

This module implements a simple RAG pipeline that:
1. Loads the HotPotQA dataset
2. Creates a vector store from the documents
3. Retrieves relevant context for questions
4. Generates answers using a language model
"""

import os
from typing import List, Dict, Any, Optional, Iterator
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
import json
from tqdm import tqdm
from dotenv import load_dotenv

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
from datasets import Dataset

# Load environment variables from .env file
load_dotenv()


class HuggingFaceInferenceLLM(LLM):
    """Custom LLM wrapper for HuggingFace Inference API.
    
    Uses direct HTTP requests for maximum compatibility with the HuggingFace
    Inference API, avoiding issues with InferenceClient API changes.
    """
    
    model: str = ""
    temperature: float = 0.7
    max_new_tokens: int = 512
    api_token: str = ""
    
    def __init__(
        self, 
        model: str, 
        temperature: float = 0.7, 
        max_new_tokens: int = 512, 
        token: Optional[str] = None,
        **kwargs
    ):
        """Initialize the HuggingFace Inference LLM.
        
        Args:
            model: HuggingFace model ID
            temperature: Sampling temperature (0.0 to 1.0)
            max_new_tokens: Maximum tokens to generate
            token: HuggingFace API token
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            api_token=token or "",
            **kwargs
        )
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens
        }
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[Any] = None,
        **kwargs
    ) -> str:
        """Call the HuggingFace Inference API using direct HTTP requests.
        
        This approach avoids compatibility issues with changing InferenceClient APIs.
        """
        try:
            import requests
            
            API_URL = f"https://api-inference.huggingface.co/models/{self.model}"
            headers = {"Authorization": f"Bearer {self.api_token}"}
            
            # Use temperature from kwargs if provided, otherwise use instance value
            temp = kwargs.get('temperature', self.temperature)
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": temp if temp > 0 else 0.01,  # Avoid 0 temperature issues
                    "return_full_text": False,
                    "do_sample": temp > 0
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                    return result[0]['generated_text']
                return str(result[0])
            elif isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text']
            else:
                return str(result)
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"\n⚠️  LLM Error: {error_msg}")
            return f"Error: {error_msg}"
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs,
    ) -> Iterator[GenerationChunk]:
        """Stream is not supported, falls back to regular call."""
        result = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
        yield GenerationChunk(text=result)


class BasicRAG:
    """
    A basic RAG system for question answering on HotPotQA dataset.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "HuggingFaceH4/zephyr-7b-beta",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3
    ):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: Name of the HuggingFace embedding model
            llm_model: Name of the HuggingFace language model
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
        """
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize components
        print("Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        self.vectorstore = None
        self.qa_chain = None
        
    def load_hotpotqa_data(
        self, 
        split: str = "train", 
        max_samples: Optional[int] = 1000,
        local_file: Optional[str] = None
    ):
        """
        Load HotPotQA dataset from HuggingFace or local file.
        
        Args:
            split: Dataset split to load ('train', 'validation')
            max_samples: Maximum number of samples to load (None for all)
            local_file: Path to local JSON file (if None, loads from HuggingFace)
            
        Returns:
            List of documents and questions
        """
        documents = []
        questions_data = []
        
        if local_file and os.path.exists(local_file):
            # Load from local file
            print(f"Loading HotPotQA dataset from local file: {local_file}...")
            with open(local_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            if max_samples:
                dataset = dataset[:max_samples]
            
            print("Processing dataset...")
            for idx, item in enumerate(tqdm(dataset)):
                # Extract context
                context_list = item['context']
                question = item['question']
                answer = item['answer']
                
                # Create documents from context
                for title, sentences in context_list:
                    if sentences:  # Check if sentences list is not empty
                        text = " ".join(sentences)
                        doc = Document(
                            page_content=text,
                            metadata={
                                "title": title,
                                "question_id": idx,
                                "source": "hotpotqa"
                            }
                        )
                        documents.append(doc)
                
                # Store question data
                questions_data.append({
                    "id": idx,
                    "question": question,
                    "answer": answer,
                    "type": item.get('type', 'unknown'),
                    "level": item.get('level', 'unknown')
                })
        else:
            # Load from HuggingFace
            print(f"Loading HotPotQA dataset from HuggingFace ({split} split)...")
            dataset = load_dataset("hotpot_qa", "distractor", split=split)
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            print("Processing dataset...")
            for idx, item in enumerate(tqdm(dataset)):
                # Extract context from supporting facts
                context_list = item['context']
                question = item['question']
                answer = item['answer']
                
                # Create documents from context
                titles = context_list['title']
                sentences_list = context_list['sentences']
                
                for title, sentences in zip(titles, sentences_list):
                    if sentences:  # Check if sentences list is not empty
                        text = " ".join(sentences)
                        doc = Document(
                            page_content=text,
                            metadata={
                                "title": title,
                                "question_id": idx,
                                "source": "hotpotqa"
                            }
                        )
                        documents.append(doc)
                
                # Store question data
                questions_data.append({
                    "id": idx,
                    "question": question,
                    "answer": answer,
                    "type": item.get('type', 'unknown'),
                    "level": item.get('level', 'unknown')
                })
        
        print(f"Loaded {len(documents)} documents and {len(questions_data)} questions")
        return documents, questions_data
    
    def create_vectorstore(self, documents: List[Document], save_path: Optional[str] = None):
        """
        Create a vector store from documents.
        
        Args:
            documents: List of Document objects
            save_path: Optional path to save the vectorstore
        """
        print("Splitting documents into chunks...")
        splits = self.text_splitter.split_documents(documents)
        print(f"Created {len(splits)} chunks from {len(documents)} documents")
        
        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        if save_path:
            print(f"Saving vector store to {save_path}...")
            self.vectorstore.save_local(save_path)
        
        print("Vector store created successfully!")
    
    def load_vectorstore(self, load_path: str):
        """
        Load a pre-existing vector store.
        
        Args:
            load_path: Path to the saved vectorstore
        """
        print(f"Loading vector store from {load_path}...")
        self.vectorstore = FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully!")
    
    def setup_qa_chain(self, huggingface_api_token: Optional[str] = None, use_gemini: bool = False):
        """
        Set up the QA chain with retriever and LLM.
        
        Args:
            huggingface_api_token: HuggingFace API token (optional, uses env var if not provided)
            use_gemini: If True, uses Google Gemini instead of HuggingFace (recommended)
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_vectorstore first.")
        
        print("Setting up QA chain...")
        
        # Set up retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        # Set up LLM
        if use_gemini:
            # Use Google Gemini (more reliable than HuggingFace free tier)
            print("Using Google Gemini for generation...")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not found. Please set it:\n"
                    "Windows: $env:GOOGLE_API_KEY='your-key'\n"
                    "Linux/Mac: export GOOGLE_API_KEY='your-key'\n"
                    "Get your key at: https://makersuite.google.com/app/apikey"
                )
            
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=google_api_key,
                temperature=0.7
            )
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
                temperature=0.7,
                max_new_tokens=512,
                token=os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
            )
        
        # Create custom prompt
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
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
        """
        Query the RAG system with a question.
        
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
        """
        Query the RAG system with multiple questions.
        
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
        """
        Evaluate a single question against ground truth.
        
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
    
    def _create_ragas_llm_and_embeddings(self, use_openai: bool = False):
        """
        Create LLM and embeddings wrapped for RAGAS evaluation.
        
        Args:
            use_openai: If True, uses OpenAI. If False, uses Google Gemini.
            
        Returns:
            Tuple of (ragas_llm, ragas_embeddings)
        """
        if use_openai:
            # OpenAI is used by default in RAGAS
            return None, None
        
        # Use Google Gemini
        print("Using Google Gemini for RAGAS evaluation (requires GOOGLE_API_KEY)")
        
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        
        # Check for Google API key
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it:\n"
                "Windows: $env:GOOGLE_API_KEY='your-key'\n"
                "Linux/Mac: export GOOGLE_API_KEY='your-key'\n"
                "Get your key at: https://makersuite.google.com/app/apikey"
            )
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            # Create Gemini LLM with temperature support
            # langchain-google-genai 4.x properly supports temperature
            evaluator_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=google_api_key,
                temperature=0
            )
            
        except ImportError:
            raise ImportError(
                "langchain-google-genai not installed. Install it with:\n"
                "pip install langchain-google-genai>=2.0.0"
            )
        
        # Wrap for RAGAS
        ragas_llm = LangchainLLMWrapper(evaluator_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        return ragas_llm, ragas_embeddings
    
    def evaluate_with_ragas(
        self, 
        questions_data: List[Dict[str, Any]], 
        metrics: Optional[List[Any]] = None,
        max_samples: Optional[int] = None,
        use_openai: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system using RAGAS framework.
        
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
        
        # Get RAGAS LLM and embeddings
        ragas_llm, ragas_embeddings = self._create_ragas_llm_and_embeddings(use_openai)
        
        # Default to all metrics if none specified
        if metrics is None:
            metrics = [
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
                answer_correctness,
                answer_similarity
            ]
        
        # Limit samples if specified
        if max_samples:
            questions_data = questions_data[:max_samples]
        
        print(f"\nEvaluating {len(questions_data)} samples with RAGAS...")
        print(f"Metrics: {[m.name for m in metrics]}")
        
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
        
        # Convert to HuggingFace Dataset format
        dataset = Dataset.from_dict(ragas_data)
        
        print("\nRunning RAGAS evaluation...")
        
        # Run RAGAS evaluation with configured LLM
        if use_openai:
            results = evaluate(dataset, metrics=metrics)
        else:
            # Use Google Gemini for evaluation
            results = evaluate(
                dataset, 
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings
            )
        
        # Convert results to dictionary format
        evaluation_results = {
            "overall_scores": {},
            "per_sample_scores": []
        }
        
        # Extract overall scores - RAGAS 0.2.x returns dict-like results
        if hasattr(results, 'items'):
            # RAGAS 0.2.x style
            for key, value in results.items():
                if key not in ['question', 'answer', 'contexts', 'ground_truth']:
                    evaluation_results["overall_scores"][key] = value
        else:
            # RAGAS 0.1.x style
            for metric in metrics:
                metric_name = metric.name
                if hasattr(results, metric_name):
                    evaluation_results["overall_scores"][metric_name] = getattr(results, metric_name)
        
        # Extract per-sample scores if available
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            for idx, row in df.iterrows():
                sample_scores = {
                    "question": ragas_data["question"][idx],
                    "answer": ragas_data["answer"][idx],
                    "ground_truth": ragas_data["ground_truth"][idx],
                }
                # Add metric scores
                for metric in metrics:
                    if metric.name in df.columns:
                        sample_scores[metric.name] = row[metric.name]
                
                evaluation_results["per_sample_scores"].append(sample_scores)
        
        return evaluation_results
    
    def evaluate_batch_with_ragas(
        self,
        results: List[Dict[str, Any]],
        metrics: Optional[List[Any]] = None,
        use_openai: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate pre-computed RAG results using RAGAS framework.
        
        Args:
            results: List of dicts containing question, predicted_answer, 
                    ground_truth, and source_documents
            metrics: List of RAGAS metrics to use. If None, uses all available metrics.
            use_openai: If True, uses OpenAI for evaluation. If False, uses Google Gemini (default).
            
        Returns:
            Dictionary containing RAGAS evaluation results
        """
        # Get RAGAS LLM and embeddings
        ragas_llm, ragas_embeddings = self._create_ragas_llm_and_embeddings(use_openai)
        
        # Default to all metrics if none specified
        if metrics is None:
            metrics = [
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
                answer_correctness,
                answer_similarity
            ]
        
        print(f"\nEvaluating {len(results)} pre-computed results with RAGAS...")
        print(f"Metrics: {[m.name for m in metrics]}")
        
        # Prepare data for RAGAS
        ragas_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for result in results:
            # Extract contexts from source documents
            contexts = [doc['content'] for doc in result.get('source_documents', [])]
            
            ragas_data["question"].append(result['question'])
            ragas_data["answer"].append(result['predicted_answer'])
            ragas_data["contexts"].append(contexts)
            ragas_data["ground_truth"].append(result['ground_truth'])
        
        # Convert to HuggingFace Dataset format
        dataset = Dataset.from_dict(ragas_data)
        
        print("\nRunning RAGAS evaluation...")
        
        # Run RAGAS evaluation with configured LLM
        if use_openai:
            eval_results = evaluate(dataset, metrics=metrics)
        else:
            eval_results = evaluate(
                dataset, 
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings
            )
        
        # Convert results to dictionary format
        evaluation_results = {
            "overall_scores": {},
            "per_sample_scores": []
        }
        
        # Extract overall scores - RAGAS 0.2.x returns dict-like results
        if hasattr(eval_results, 'items'):
            # RAGAS 0.2.x style
            for key, value in eval_results.items():
                if key not in ['question', 'answer', 'contexts', 'ground_truth']:
                    evaluation_results["overall_scores"][key] = value
        else:
            # RAGAS 0.1.x style
            for metric in metrics:
                metric_name = metric.name
                if hasattr(eval_results, metric_name):
                    evaluation_results["overall_scores"][metric_name] = getattr(eval_results, metric_name)
        
        # Extract per-sample scores if available
        if hasattr(eval_results, 'to_pandas'):
            df = eval_results.to_pandas()
            for idx, row in df.iterrows():
                sample_scores = {
                    "question": ragas_data["question"][idx],
                    "answer": ragas_data["answer"][idx],
                    "ground_truth": ragas_data["ground_truth"][idx],
                }
                # Add metric scores
                for metric in metrics:
                    if metric.name in df.columns:
                        sample_scores[metric.name] = row[metric.name]
                
                evaluation_results["per_sample_scores"].append(sample_scores)
        
        return evaluation_results


def main():
    """
    Main function - Use run_rag.py instead for better control.
    """
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
