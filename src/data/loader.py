"""
Data loading and preprocessing for RAG training.

This module handles loading HotpotQA dataset from CSV files and preparing
training examples for retriever and generator training.
"""

import ast
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm


@dataclass
class TrainExample:
    """Training example for generator."""
    question: str
    answer: str
    contexts: Optional[Sequence[str]] = None  # Pre-computed contexts (optional)


@dataclass
class RetrieverExample:
    """Training example for retriever."""
    question: str
    positive_passage: str
    negative_passages: Optional[Sequence[str]] = None


class HotpotQALoader:
    """
    Loader for HotpotQA dataset in CSV format.
    
    Expected CSV format:
        - question: str
        - answer: str
        - context: JSON string with {"title": [...], "sentences": [[...], [...]]}
        - supporting_facts: JSON string with {"title": [...]}
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self.df_train: Optional[pd.DataFrame] = None
        self.df_valid: Optional[pd.DataFrame] = None
        self.corpus_texts: List[str] = []
        self.doc_titles: List[str] = []
    
    def load_data(
        self, 
        train_path: str, 
        valid_path: str,
        max_train_samples: Optional[int] = None,
        max_valid_samples: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load HotpotQA data from CSV files.
        
        Args:
            train_path: Path to training CSV file
            valid_path: Path to validation CSV file
            max_train_samples: Maximum number of training samples (None = all)
            max_valid_samples: Maximum number of validation samples (None = all)
            
        Returns:
            Tuple of (train_df, valid_df)
        """
        print(f"Loading HotpotQA data...")
        
        # Load and parse training data
        self.df_train = self._parse_csv(train_path)
        if max_train_samples:
            self.df_train = self.df_train.head(max_train_samples)
        
        # Load and parse validation data
        self.df_valid = self._parse_csv(valid_path)
        if max_valid_samples:
            self.df_valid = self.df_valid.head(max_valid_samples)
        
        print(f"Loaded {len(self.df_train)} training samples")
        print(f"Loaded {len(self.df_valid)} validation samples")
        
        return self.df_train, self.df_valid
    
    @staticmethod
    def _parse_csv(filepath: str) -> pd.DataFrame:
        """
        Parse HotpotQA CSV file.
        
        The context column contains JSON strings that need to be parsed
        into structured documents with title and text.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Parsed DataFrame with 'docs' column added
        """
        df = pd.read_csv(filepath)
        
        def join_context(ctx_str):
            """Parse context JSON string into list of documents."""
            try:
                ctx = ast.literal_eval(ctx_str)
                docs = []
                
                # Handle both dict and list formats
                if isinstance(ctx, dict):
                    titles = ctx.get("title", [])
                    sentences_list = ctx.get("sentences", [])
                elif isinstance(ctx, list):
                    # Format: [[title, [sentences]], ...]
                    titles = [item[0] for item in ctx]
                    sentences_list = [item[1] for item in ctx]
                else:
                    return []
                
                # Create document list
                for title, sentences in zip(titles, sentences_list):
                    if sentences and len(sentences) > 0:
                        text = " ".join(sentences)
                        docs.append({"title": title, "text": text})
                
                return docs
            except Exception as e:
                print(f"Warning: Error parsing context: {e}")
                return []
        
        # Parse context into structured documents
        df["docs"] = df["context"].apply(join_context)
        
        return df
    
    def build_corpus(
        self, 
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Build corpus from DataFrame by flattening all documents.
        
        Args:
            df: DataFrame to build corpus from (default: training data)
            
        Returns:
            Tuple of (corpus_texts, doc_titles)
        """
        df = df or self.df_train
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        corpus = []
        for docs in df["docs"]:
            corpus.extend(docs)
        
        # Extract texts and titles
        self.corpus_texts = [doc["text"] for doc in corpus]
        self.doc_titles = [doc["title"] for doc in corpus]
        
        print(f"Built corpus: {len(self.corpus_texts)} passages from {len(df)} questions")
        
        return self.corpus_texts, self.doc_titles
    
    def create_retriever_examples(
        self,
        df: Optional[pd.DataFrame] = None,
        max_examples: Optional[int] = None
    ) -> List[RetrieverExample]:
        """
        Create training examples for retriever.
        
        Each example consists of:
        - question: The query
        - positive_passage: A relevant passage (from supporting_facts)
        - negative_passages: Irrelevant passages (optional)
        
        Args:
            df: DataFrame to create examples from (default: training data)
            max_examples: Maximum number of examples to create
            
        Returns:
            List of RetrieverExample objects
        """
        df = df or self.df_train
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        examples = []
        
        print("Creating retriever training examples...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            if max_examples and len(examples) >= max_examples:
                break
            
            question = row["question"]
            docs = row["docs"]
            
            # Parse supporting facts to get positive document titles
            try:
                supporting_facts = row["supporting_facts"]
                if isinstance(supporting_facts, str):
                    supporting_facts = ast.literal_eval(supporting_facts)
                
                if isinstance(supporting_facts, dict):
                    pos_titles = supporting_facts.get("title", [])
                elif isinstance(supporting_facts, list):
                    # Format: [[title, sent_idx], ...]
                    pos_titles = list(set([item[0] for item in supporting_facts]))
                else:
                    continue
            except Exception as e:
                continue
            
            # Separate positive and negative documents
            pos_docs = [d["text"] for d in docs if d["title"] in pos_titles]
            neg_docs = [d["text"] for d in docs if d["title"] not in pos_titles]
            
            # Skip if no positive or negative examples
            if not pos_docs or not neg_docs:
                continue
            
            # Create example with random positive passage
            positive_passage = random.choice(pos_docs)
            examples.append(
                RetrieverExample(
                    question=question,
                    positive_passage=positive_passage,
                    negative_passages=neg_docs if neg_docs else None
                )
                )
        
        print(f"Created {len(examples)} retriever training examples")
        return examples
    
    def create_generator_examples(
        self,
        df: Optional[pd.DataFrame] = None,
        max_examples: Optional[int] = None,
        include_contexts: bool = False
    ) -> List[TrainExample]:
        """
        Create training examples for generator.
        
        Each example consists of:
        - question: The query
        - answer: The target answer
        - contexts: Pre-computed contexts (optional)
        
        Args:
            df: DataFrame to create examples from (default: training data)
            max_examples: Maximum number of examples to create
            include_contexts: Whether to pre-compute contexts
            
        Returns:
            List of TrainExample objects
        """
        df = df or self.df_train
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        examples = []
        
        print("Creating generator training examples...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            if max_examples and len(examples) >= max_examples:
                break
            
            question = row["question"]
            answer = row["answer"]
            
            # Optionally include pre-computed contexts
            contexts = None
            if include_contexts:
                docs = row["docs"]
                contexts = [d["text"] for d in docs]
            
            examples.append(
                TrainExample(
                    question=question,
                    answer=answer,
                    contexts=contexts
                )
                )
        
        print(f"Created {len(examples)} generator training examples")
        return examples
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the loaded data.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        if self.df_train is not None:
            stats["train_samples"] = len(self.df_train)
            stats["train_avg_docs_per_question"] = self.df_train["docs"].apply(len).mean()
        
        if self.df_valid is not None:
            stats["valid_samples"] = len(self.df_valid)
            stats["valid_avg_docs_per_question"] = self.df_valid["docs"].apply(len).mean()
        
        if self.corpus_texts:
            stats["corpus_size"] = len(self.corpus_texts)
            stats["avg_passage_length"] = sum(len(t.split()) for t in self.corpus_texts) / len(self.corpus_texts)
        
        return stats
