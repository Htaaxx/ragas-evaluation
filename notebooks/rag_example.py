"""
Example usage of RAGSystem with HotpotQA

Setup:
1. Tokens already configured in .env file ✅
   - HUGGINGFACEHUB_API_TOKEN=hf_JeJMdWcfq...
   - HF_TOKEN=hf_JeJMdWcfq...

2. Have data files (optional for some examples):
   - train.csv (with columns: question, context, supporting_facts)
   - valid.csv (same structure)

3. Install dependencies:
   pip install torch transformers sentence-transformers faiss-cpu pandas tqdm
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import RAGSystem
from rag import RAGSystem, TrainExample


def _load_hf_token():
	"""Load HF token from environment (from .env or system)"""
	token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
	if token:
		print(f"✅ HF Token loaded: {token[:30]}...")
		return token
	print("⚠️  No HF token found (optional for public models)")
	return None


def example_basic_qa():
	"""Simple example: Build index and answer questions"""
	print("\n" + "="*70)
	print("EXAMPLE 1: Basic QA (no training)")
	print("="*70)

	docs = [
		"The Eiffel Tower is in Paris, France.",
		"The Colosseum is in Rome, Italy.",
		"The Statue of Liberty is in New York, USA.",
		"Paris is known for art and culture.",
		"Rome has many ancient ruins.",
	]

	# Initialize RAG with HF token
	hf_token = _load_hf_token()
	rag = RAGSystem(output_dir="./rag_output", hf_token=hf_token)

	# Build and save index
	rag.index_corpus(docs)
	rag.save_index()

	# Test retrieval
	print("\n🔍 Retrieval Test:")
	query = "Where is the Eiffel Tower?"
	results = rag.retrieve(query, top_k=2)
	print(f"Query: {query}")
	for i, doc in enumerate(results, 1):
		print(f"  {i}. {doc}")

	# Test QA
	print("\n🤖 QA Test:")
	answer = rag.answer(query, top_k=2)
	print(f"Answer: {answer}")


def example_retriever_training():
	"""Example: Train retriever on HotpotQA"""
	print("\n" + "="*70)
	print("EXAMPLE 2: Retriever Training")
	print("="*70)

	# Get HF token from environment (from .env)
	hf_token = _load_hf_token()

	# Initialize
	rag = RAGSystem(
		encoder_model="sentence-transformers/all-MiniLM-L6-v2",
		output_dir="./rag_output",
		hf_token=hf_token,
	)

	# Load HotpotQA data
	print("\n📂 Loading HotpotQA data...")
	try:
		rag.load_hotpotqa_data(
			train_path="data/hotpot_qa/train.csv",
			valid_path="data/hotpot_qa/valid.csv",
		)
	except FileNotFoundError:
		print("⚠️  Data not found. Skipping training example.")
		return

	# Build corpus
	rag.build_corpus()

	# Create training data
	print("\n📝 Creating retriever training data...")
	rag.create_retriever_train_loader(batch_size=16)

	# Train retriever
	print("\n🚀 Training Retriever...")
	rag.train_retriever(
		epochs=3,
		lr=2e-5,
		patience=2,
		save_name="retriever_hotpotqa",
	)

	# Evaluate retriever
	print("\n📊 Evaluating Retriever...")
	metrics = rag.evaluate_retriever(top_k=5, cache_name="corpus_hotpotqa")

	print(f"\n✅ Training Results:")
	print(f"  Top-1 Accuracy: {metrics['top1_acc']:.4f}")
	print(f"  Top-3 Accuracy: {metrics['top3_acc']:.4f}")
	print(f"  Top-5 Accuracy: {metrics['top5_acc']:.4f}")
	print(f"  MRR: {metrics['mrr']:.4f}")

	# Index corpus
	rag.index_corpus(rag.corpus_texts)
	rag.save_index()


def example_generator_training():
	"""Example: Train generator on HotpotQA"""
	print("\n" + "="*70)
	print("EXAMPLE 3: Generator Training")
	print("="*70)

	hf_token = _load_hf_token()

	# Initialize
	rag = RAGSystem(
		encoder_model="sentence-transformers/all-MiniLM-L6-v2",
		generator_model="google/flan-t5-base",
		models_dir="../models"
		hf_token=hf_token,
	)

	# Prepare data
	print("\n📂 Loading HotpotQA data...")
	try:
		rag.load_hotpotqa_data(
			train_path="data/hotpot_qa/train.csv",
			valid_path="data/hotpot_qa/valid.csv",
		)
	except FileNotFoundError:
		print("⚠️  Data not found. Skipping generator training example.")
		return

	rag.build_corpus()

	# Build index first (for retrieval during training)
	print("\n🔨 Building index...")
	rag.index_corpus(rag.corpus_texts)

	# Prepare training examples
	print("\n📝 Preparing training examples...")
	train_examples = []
	for i, (_, row) in enumerate(rag.df_train.iterrows()):
		if i >= 100:  # Use first 100 for demo
			break
		train_examples.append(
			TrainExample(
				question=row["question"],
				answer=row["answer"],  # assuming "answer" column exists
			)
		)

	if not train_examples:
		print("⚠️  No training examples found. Make sure 'answer' column exists.")
		return

	# Train generator
	print(f"\n🚀 Training Generator on {len(train_examples)} examples...")
	rag.train_generator(
		train_examples=train_examples,
		batch_size=4,
		epochs=1,
		lr=5e-5,
		top_k=5,
	)

	print("\n✅ Generator training complete!")


def example_end_to_end():
	"""Example: Full RAG pipeline (retriever + generator + QA)"""
	print("\n" + "="*70)
	print("EXAMPLE 4: End-to-End RAG")
	print("="*70)

	hf_token = _load_hf_token()

	rag = RAGSystem(
		encoder_model="sentence-transformers/all-MiniLM-L6-v2",
		generator_model="google/flan-t5-base",
		output_dir="./rag_output",
		hf_token=hf_token,
	)

	# Load and process data
	print("\n📂 Loading data...")
	try:
		rag.load_hotpotqa_data(
			train_path="data/hotpot_qa/train.csv",
			valid_path="data/hotpot_qa/valid.csv",
		)
		rag.build_corpus()
	except FileNotFoundError:
		print("⚠️  Data not found. Using demo documents instead...")
		demo_docs = [
			"Paris is the capital of France.",
			"The Eiffel Tower is located in Paris.",
			"France is in Western Europe.",
		]
		rag.corpus_texts = demo_docs

	# Step 1: Train retriever
	print("\n1️⃣  Training retriever...")
	rag.create_retriever_train_loader(batch_size=16)
	rag.train_retriever(epochs=2, lr=2e-5, patience=1, save_name="retriever_e2e")

	# Step 2: Build index
	print("\n2️⃣  Building index...")
	rag.index_corpus(rag.corpus_texts)
	rag.save_index()

	# Step 3: Evaluate retriever
	print("\n3️⃣  Evaluating retriever...")
	try:
		metrics = rag.evaluate_retriever(top_k=5)
		print(f"Retriever MRR: {metrics['mrr']:.4f}")
	except Exception as e:
		print(f"⚠️  Evaluation skipped: {e}")

	# Step 4: Answer questions
	print("\n4️⃣  Answering questions...")
	questions = [
		"What is the capital of France?",
		"Where is the Eiffel Tower?",
	]

	for q in questions:
		print(f"\n  Q: {q}")
		try:
			answer = rag.answer(q, top_k=2)
			print(f"  A: {answer}")
		except Exception as e:
			print(f"  A: (Error: {e})")

	print("\n✅ End-to-end pipeline complete!")


if __name__ == "__main__":
	# Run examples
	print("\n🎯 RAG System Examples\n")

	# Basic QA - always works
	example_basic_qa()

	# Commented out - requires data files
	# example_retriever_training()
	# example_generator_training()
	# example_end_to_end()

	print("\n" + "="*70)
	print("📚 To run full examples, prepare:")
	print("   - data/hotpot_qa/train.csv")
	print("   - data/hotpot_qa/valid.csv")
	print("   - export HF_TOKEN=<your-token> (if using private models)")
	print("="*70 + "\n")
