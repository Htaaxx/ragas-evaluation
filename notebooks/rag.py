"""
Comprehensive RAG implementation:
- Retriever: fine-tune sentence-transformers on question-passage pairs
- Generator: fine-tune seq2seq model with retrieved contexts
- Index: build/load FAISS vector index
- QA: retrieve-and-generate pipeline
- Model Caching: Auto download & cache models locally in ../models/

Dependencies (pip):
	torch transformers sentence-transformers faiss-cpu pandas tqdm
"""

from __future__ import annotations

import ast
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import InputExample, SentenceTransformer, util
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
	AdamW,
	AutoModelForSeq2SeqLM,
	AutoTokenizer,
	get_linear_schedule_with_warmup,
)


@dataclass
class TrainExample:
	question: str
	answer: str
	contexts: Sequence[str] | None = None  # Optional pre-computed contexts


@dataclass
class RetrieverExample:
	question: str
	positive_passage: str
	negative_passages: Sequence[str] | None = None


class _RetrieverDataset(Dataset):
	def __init__(self, examples: Sequence[RetrieverExample]):
		self.examples = examples

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, idx: int) -> RetrieverExample:
		return self.examples[idx]


class _TrainDataset(Dataset):
	def __init__(self, examples: Sequence[TrainExample]):
		self.examples = examples

	def __len__(self) -> int:  # pragma: no cover - trivial
		return len(self.examples)

	def __getitem__(self, idx: int) -> TrainExample:  # pragma: no cover - trivial
		return self.examples[idx]


class RAGSystem:
	"""End-to-end RAG: retriever training + generator training + QA
	
	Features:
	- Auto-cache models locally in ../models/
	- Check local before downloading from HuggingFace
	- Save trained models locally
	- Load models from local cache for QA
	"""

	def __init__(
		self,
		encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
		generator_model: str = "google/flan-t5-base",
		index_dir: str | Path | None = None,
		output_dir: str | Path = "./rag_output",
		models_dir: str | Path = "../models",
		device: str | None = None,
		hf_token: str | None = None,
	) -> None:
		"""
		Args:
			encoder_model: HF model ID for retriever
			generator_model: HF model ID for generator
			index_dir: Directory with pre-built FAISS index
			models_dir: Where to cache downloaded models (default: ../models) & save trained models
			device: 'cuda' or 'cpu'
			output_dir: Directory for outputs & caches
			hf_token: HuggingFace API token (for private repos)
		"""
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.models_dir = Path(models_dir)
		self.models_dir.mkdir(parents=True, exist_ok=True)
		self.hf_token = hf_token
		
		# Store model IDs
		self.encoder_model_id = encoder_model
		self.generator_model_id = generator_model

		# Disable HF repo templates check
		import transformers.utils.hub as hub
		hub.list_repo_templates = lambda *args, **kwargs: []

		# Load models with caching
		print(f"🏗️  Models directory: {self.models_dir.absolute()}")
		
		encoder_path = self._load_or_download_model(encoder_model)
		self.encoder = SentenceTransformer(str(encoder_path), device=self.device)
		
		generator_path = self._load_or_download_model(generator_model, is_generator=True)
		self.generator = AutoModelForSeq2SeqLM.from_pretrained(
			str(generator_path), token=hf_token
		).to(self.device)
		self.tokenizer = AutoTokenizer.from_pretrained(str(generator_path), token=hf_token)

		# Index
		self.index: faiss.IndexFlatIP | None = None
		self.docstore: list[str] = []
		self.doc_titles: list[str] = []  # Track document titles
		if index_dir:
			self.load_index(index_dir)

		# Data
		self.df_train: pd.DataFrame | None = None
		self.df_valid: pd.DataFrame | None = None
		self.corpus_texts: list[str] = []
		self.train_loader: DataLoader | None = None

	# ======================= MODEL CACHING =======================
	def _get_local_model_path(self, model_id: str) -> Path:
		"""Get local path for a HuggingFace model
		
		Example: 
			"sentence-transformers/all-MiniLM-L6-v2" → 
			../models/sentence-transformers--all-MiniLM-L6-v2
		"""
		return self.models_dir / model_id.replace("/", "--")
	
	def _load_or_download_model(self, model_id: str, is_generator: bool = False) -> Path:
		"""Load model from local cache or download from HuggingFace
		
		Args:
			model_id: HuggingFace model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
			is_generator: If True, use AutoModel; else use snapshot_download
			
		Returns:
			Path to model directory
		"""
		local_path = self._get_local_model_path(model_id)
		
		# Check if model exists locally
		if local_path.exists() and (list(local_path.glob("*.bin")) or list(local_path.glob("*.safetensors"))):
			print(f"✅ Loaded from cache: {local_path.name}")
			return local_path
		
		# Download from HuggingFace
		print(f"📥 Downloading {model_id}...")
		try:
			# Download directly to local_dir (no moving needed)
			snapshot_download(
				model_id,
				local_dir=str(local_path),
				local_dir_use_symlinks=False,
				token=self.hf_token,
			)
			print(f"✅ Saved to cache: {local_path.name}")
			return local_path
			
		except Exception as e:
			print(f"❌ Error downloading {model_id}: {e}")
			raise

	# ======================= DATA LOADING =======================
	def load_hotpotqa_data(self, train_path: str, valid_path: str) -> tuple:
		"""Load HotpotQA CSV files with context as JSON strings"""
		print(f"📂 Loading HotpotQA data...")
		self.df_train = self._parse_hotpotqa_csv(train_path)
		self.df_valid = self._parse_hotpotqa_csv(valid_path)
		print(f"✅ Train: {len(self.df_train)}, Valid: {len(self.df_valid)}")
		return self.df_train, self.df_valid

	@staticmethod
	def _parse_hotpotqa_csv(filepath: str) -> pd.DataFrame:
		"""Parse CSV with context as JSON string"""
		df = pd.read_csv(filepath)

		def join_context(ctx_str):
			ctx = ast.literal_eval(ctx_str)
			docs = []
			for t, sents in zip(ctx["title"], ctx["sentences"]):
				if len(sents) > 0:
					docs.append({"title": t, "text": " ".join(sents)})
			return docs

		df["docs"] = df["context"].apply(join_context)
		return df

	def build_corpus(self, df: pd.DataFrame | None = None) -> tuple[list, list]:
		"""Flatten all docs from DataFrame into corpus"""
		df = df or self.df_train
		if df is None:
			raise ValueError("⚠️ Load data first or provide DataFrame")

		corpus = []
		for docs in df["docs"]:
			corpus.extend(docs)

		self.corpus_texts = [c["text"] for c in corpus]
		self.doc_titles = [c["title"] for c in corpus]
		print(f"✅ Built corpus: {len(self.corpus_texts)} passages")
		return self.corpus_texts, self.doc_titles

	# ======================= RETRIEVER TRAINING =======================
	def create_retriever_train_loader(
		self, batch_size: int = 16, shuffle: bool = True
	) -> DataLoader:
		"""Create training pairs (question, positive_passage)"""
		if self.df_train is None:
			raise ValueError("⚠️ Load data first")

		train_examples = []
		for _, row in self.df_train.iterrows():
			ctx = row["docs"]
			try:
				pos_titles = ast.literal_eval(row["supporting_facts"])["title"]
			except:
				pos_titles = row["supporting_facts"]["title"]

			pos_docs = [d["text"] for d in ctx if d["title"] in pos_titles]
			neg_docs = [d["text"] for d in ctx if d["title"] not in pos_titles]

			if not pos_docs or not neg_docs:
				continue

			pos = random.choice(pos_docs)
			train_examples.append(InputExample(texts=[row["question"], pos]))

		self.train_loader = DataLoader(
			train_examples,
			batch_size=batch_size,
			shuffle=shuffle,
			drop_last=True,
			collate_fn=lambda x: x,
		)
		print(f"✅ Created retriever loader: {len(train_examples)} pairs")
		return self.train_loader

	def train_retriever(
		self,
		epochs: int = 5,
		lr: float = 2e-5,
		patience: int = 3,
		temperature: float = 20.0,
		accumulation_steps: int = 2,
		use_fp16: bool = True,
		save_name: str = "retriever_trained",
	) -> SentenceTransformer:
		"""Fine-tune retriever with question-passage pairs"""
		if self.train_loader is None:
			raise ValueError("⚠️ Call create_retriever_train_loader() first")

		self.encoder.to(self.device)
		optimizer = optim.AdamW(self.encoder.parameters(), lr=lr)
		loss_fn = nn.CrossEntropyLoss()
		scaler = GradScaler(enabled=use_fp16)

		best_loss = float("inf")
		patience_counter = 0

		if hasattr(self.encoder, "max_seq_length"):
			self.encoder.max_seq_length = min(self.encoder.max_seq_length, 128)

		for epoch in range(epochs):
			self.encoder.train()
			epoch_losses = []
			progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
			optimizer.zero_grad(set_to_none=True)

			for step, batch in enumerate(progress):
				questions = [ex.texts[0] for ex in batch]
				passages = [ex.texts[1] for ex in batch]

				q_features = self.encoder.tokenize(questions)
				p_features = self.encoder.tokenize(passages)

				q_features = {k: v.to(self.device) for k, v in q_features.items()}
				p_features = {k: v.to(self.device) for k, v in p_features.items()}

				with autocast(enabled=use_fp16):
					q_emb = self.encoder(q_features)["sentence_embedding"]
					p_emb = self.encoder(p_features)["sentence_embedding"]

					sim = torch.matmul(q_emb, p_emb.T) * temperature
					labels = torch.arange(sim.size(0)).to(self.device)
					loss = loss_fn(sim, labels) / accumulation_steps

				scaler.scale(loss).backward()
				if (step + 1) % accumulation_steps == 0:
					scaler.step(optimizer)
					scaler.update()
					optimizer.zero_grad(set_to_none=True)
					torch.cuda.empty_cache()

				epoch_losses.append(loss.item())
				progress.set_postfix({"loss": f"{loss.item():.4f}"})

			avg_loss = np.mean(epoch_losses)
			print(f"📉 Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

			if avg_loss < best_loss - 1e-4:
				best_loss = avg_loss
				patience_counter = 0
				save_path = self.models_dir / save_name
				self.encoder.save(str(save_path))
				print(f"✅ Saved best model → {save_path}")
			else:
				patience_counter += 1
				if patience_counter >= patience:
					print("🛑 Early stopping")
					break

		print(f"🎯 Retriever training done. Best loss={best_loss:.4f}")
		return self.encoder

	def evaluate_retriever(
		self,
		top_k: int = 5,
		rebuild_cache: bool = False,
		batch_size: int = 64,
		cache_name: str = "corpus_embeds",
	) -> dict:
		"""Evaluate retriever on validation set"""
		if self.df_valid is None:
			raise ValueError("⚠️ Load data first")

		cache_path = self.output_dir / f"{cache_name}.pt"

		# Build corpus
		all_titles, all_texts = [], []
		for row in self.df_valid["docs"]:
			for d in row:
				all_titles.append(d["title"])
				all_texts.append(d["text"])

		unique_corpus = dict(zip(all_titles, all_texts))
		corpus_titles = list(unique_corpus.keys())
		corpus_texts = list(unique_corpus.values())

		# Encode corpus
		if not rebuild_cache and cache_path.exists():
			print(f"✅ Loading corpus embeddings from {cache_path}")
			corpus_embs = torch.load(cache_path, map_location=self.device)
		else:
			print(f"📦 Encoding {len(corpus_texts)} passages...")
			self.encoder.eval()
			corpus_embs = self.encoder.encode(
				corpus_texts,
				batch_size=batch_size,
				convert_to_tensor=True,
				normalize_embeddings=True,
			)
			torch.save(corpus_embs, cache_path)

		# Evaluate
		top1_hits, top3_hits, top5_hits, mrrs = [], [], [], []
		self.encoder.eval()

		for _, row in tqdm(
			self.df_valid.iterrows(), total=len(self.df_valid), desc="Evaluating"
		):
			question = row["question"]
			try:
				gold_titles = set(ast.literal_eval(row["supporting_facts"])["title"])
			except:
				gold_titles = set(row["supporting_facts"]["title"])

			q_emb = self.encoder.encode(
				question,
				convert_to_tensor=True,
				normalize_embeddings=True,
			)
			sims = util.cos_sim(q_emb, corpus_embs)[0]
			top_idx = torch.topk(sims, k=top_k).indices.tolist()
			ranked_titles = [corpus_titles[i] for i in top_idx]

			top1_hits.append(any(t in gold_titles for t in ranked_titles[:1]))
			top3_hits.append(any(t in gold_titles for t in ranked_titles[:3]))
			top5_hits.append(any(t in gold_titles for t in ranked_titles[:5]))

			first_hit = next(
				(i for i, t in enumerate(ranked_titles) if t in gold_titles), None
			)
			if first_hit is not None:
				mrrs.append(1.0 / (first_hit + 1))

		metrics = {
			"top1_acc": np.mean(top1_hits),
			"top3_acc": np.mean(top3_hits),
			"top5_acc": np.mean(top5_hits),
			"mrr": np.mean(mrrs),
		}

		print(f"\n📊 Retriever Metrics:")
		for k, v in metrics.items():
			print(f"  {k}: {v:.4f}")

		return metrics

	# ======================= INDEXING =======================
	def index_corpus(self, documents: Sequence[str], batch_size: int = 64) -> None:
		"""Build FAISS index from documents"""
		self.docstore = list(documents)
		embeddings = []
		for start in range(0, len(self.docstore), batch_size):
			chunk = self.docstore[start : start + batch_size]
			embs = self.encoder.encode(
				chunk, batch_size=batch_size, convert_to_numpy=True
			)
			embeddings.append(embs)
		matrix = np.vstack(embeddings).astype("float32")
		faiss.normalize_L2(matrix)
		self.index = faiss.IndexFlatIP(matrix.shape[1])
		self.index.add(matrix)
		print(f"✅ Built FAISS index with {len(self.docstore)} documents")

	def save_index(self, index_dir: str | Path = None) -> None:
		"""Save index and docstore"""
		if self.index is None:
			raise RuntimeError("Index not built")
		index_dir = Path(index_dir or self.output_dir / "index")
		index_dir.mkdir(parents=True, exist_ok=True)
		faiss.write_index(self.index, str(index_dir / "index.faiss"))
		meta = {"docstore": self.docstore, "doc_titles": self.doc_titles}
		with open(index_dir / "meta.json", "w", encoding="utf-8") as f:
			json.dump(meta, f, ensure_ascii=False, indent=2)
		print(f"✅ Saved index to {index_dir}")

	def load_index(self, index_dir: str | Path) -> None:
		"""Load index and docstore"""
		index_dir = Path(index_dir)
		self.index = faiss.read_index(str(index_dir / "index.faiss"))
		with open(index_dir / "meta.json", "r", encoding="utf-8") as f:
			meta = json.load(f)
		self.docstore = meta["docstore"]
		self.doc_titles = meta.get("doc_titles", [])
		print(f"✅ Loaded index with {len(self.docstore)} documents")

	# ======================= GENERATOR TRAINING =======================
	def train_generator(
		self,
		train_examples: Sequence[TrainExample],
		batch_size: int = 4,
		epochs: int = 3,
		lr: float = 5e-5,
		warmup_ratio: float = 0.1,
		gradient_accumulation: int = 1,
		max_input_tokens: int = 512,
		max_target_tokens: int = 128,
		top_k: int = 5,
		save_name: str = "generator_trained",
	) -> None:
		"""Fine-tune generator with retrieved contexts"""
		if self.index is None:
			raise RuntimeError("Index not built; call index_corpus first")

		dataset = _TrainDataset(train_examples)
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
		optimizer = AdamW(self.generator.parameters(), lr=lr)

		total_steps = math.ceil(len(loader) * epochs / gradient_accumulation)
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=int(total_steps * warmup_ratio),
			num_training_steps=total_steps,
		)

		self.generator.train()
		step = 0
		for epoch in range(epochs):
			progress = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
			for batch in progress:
				batch_inputs = []
				batch_targets = []
				for ex in batch:
					contexts = (
						list(ex.contexts)
						if ex.contexts
						else self.retrieve(ex.question, top_k)
					)
					prompt = self._build_prompt(ex.question, contexts)
					batch_inputs.append(prompt)
					batch_targets.append(ex.answer)

				inputs = self.tokenizer(
					batch_inputs,
					padding=True,
					truncation=True,
					max_length=max_input_tokens,
					return_tensors="pt",
				).to(self.device)
				labels = self.tokenizer(
					batch_targets,
					padding=True,
					truncation=True,
					max_length=max_target_tokens,
					return_tensors="pt",
				).input_ids.to(self.device)
				labels[labels == self.tokenizer.pad_token_id] = -100

				outputs = self.generator(**inputs, labels=labels)
				loss = outputs.loss / gradient_accumulation
				loss.backward()

				if (step + 1) % gradient_accumulation == 0:
					optimizer.step()
					scheduler.step()
					optimizer.zero_grad()

				progress.set_postfix({"loss": f"{loss.item():.4f}"})
				step += 1

		self.generator.eval()
		save_path = self.models_dir / save_name
		self.generator.save_pretrained(save_path)
		self.tokenizer.save_pretrained(save_path)
		print(f"✅ Saved trained generator to {save_path}")

	# ======================= RETRIEVAL & QA =======================
	def retrieve(self, query: str, top_k: int = 5) -> list[str]:
		"""Retrieve top-k documents"""
		if self.index is None:
			raise RuntimeError("Index not built")
		q_emb = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
		faiss.normalize_L2(q_emb)
		scores, idxs = self.index.search(q_emb, top_k)
		results = [self.docstore[i] for i in idxs[0] if i < len(self.docstore)]
		return results

	def answer(
		self,
		question: str,
		top_k: int = 5,
		max_new_tokens: int = 64,
		temperature: float | None = None,
	) -> str:
		"""Answer question via retrieve-and-generate"""
		contexts = self.retrieve(question, top_k=top_k)
		prompt = self._build_prompt(question, contexts)
		inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
		with torch.no_grad():
			outputs = self.generator.generate(
				**inputs,
				max_new_tokens=max_new_tokens,
				do_sample=temperature is not None,
				temperature=temperature or 1.0,
			)
		return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

	# ======================= HELPERS =======================
	@staticmethod
	def _build_prompt(question: str, contexts: Sequence[str]) -> str:
		joined_ctx = "\n".join(f"- {c}" for c in contexts)
		return f"Use the context to answer.\nContext:\n{joined_ctx}\nQuestion: {question}\nAnswer:"


def _demo() -> None:
	"""Demo: retriever training + QA on HotpotQA"""
	import os

	# Setup HF token (optional)
	hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

	# Initialize (models auto-cached to ../models/)
	print("\n" + "=" * 60)
	print("🔍 RAG System Demo with Model Caching")
	print("=" * 60)
	
	rag = RAGSystem(
		output_dir="./rag_output",
		models_dir="../models",  # Models cached here
		hf_token=hf_token
	)

	# Example: Simple QA without training
	docs = [
		"The Eiffel Tower is in Paris, France.",
		"The Colosseum is in Rome, Italy.",
		"The Statue of Liberty is in New York, USA.",
		"Paris is known for art and culture.",
		"Rome has many ancient ruins.",
	]

	# Build index
	rag.index_corpus(docs)
	rag.save_index("./rag_output/index")

	# Test retrieval
	print("\n📝 Testing Retrieval:")
	query = "Where is the Eiffel Tower?"
	results = rag.retrieve(query, top_k=3)
	print(f"Query: {query}")
	print(f"Results: {results}")

	# Test QA
	print("\n🤖 Testing QA:")
	answer = rag.answer(query, top_k=2)
	print(f"Answer: {answer}")


def _demo_training() -> None:
	"""Demo: Train retriever on HotpotQA (requires data)"""
	import os

	hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

	rag = RAGSystem(
		encoder_model="sentence-transformers/all-MiniLM-L6-v2",
		output_dir="./rag_output",
		models_dir="../models",
		hf_token=hf_token,
	)

	# Load HotpotQA
	print("\n" + "=" * 60)
	print("📚 RAG Training Demo")
	print("=" * 60)

	try:
		rag.load_hotpotqa_data("train.csv", "valid.csv")
		rag.build_corpus()
		rag.create_retriever_train_loader(batch_size=16)

		# Train retriever
		print("\n🚀 Training Retriever...")
		rag.train_retriever(epochs=3, lr=2e-5, patience=2)

		# Evaluate
		print("\n📊 Evaluating Retriever...")
		metrics = rag.evaluate_retriever(top_k=5)

		# Index corpus for QA
		rag.index_corpus(rag.corpus_texts)
		rag.save_index("./rag_output/index")

	except FileNotFoundError as e:
		print(f"⚠️  Data files not found: {e}")
		print("   Make sure train.csv and valid.csv are in current directory")


if __name__ == "__main__":
	_demo()
