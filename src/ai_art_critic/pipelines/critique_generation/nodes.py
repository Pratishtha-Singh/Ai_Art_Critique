"""
This module contains critique generation nodes for the AI Art Critic project.
"""

import faiss
import json
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from transformers import BertTokenizer, BertModel
import torch
from langchain_community.llms import GPT4All
from huggingface_hub import hf_hub_download
from func_timeout import func_timeout, FunctionTimedOut

logger = logging.getLogger(__name__)


class ArtCriticGenerator:
    """
    A class to handle art critique generation using retrieval-augmented generation.
    """

    def __init__(self, index_path: str, metadata_path: str, model_repo: str = "microsoft/Phi-3-mini-4k-instruct-gguf",
                 model_filename: str = "Phi-3-mini-4k-instruct-q4.gguf", max_tokens: int = 200):
        """
        Initialize the critic generator.

        Args:
            index_path: Path to Faiss index file
            metadata_path: Path to metadata JSON file
            model_repo: Hugging Face repo for the model
            model_filename: Model filename
            max_tokens: Max tokens for generation
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model_repo = model_repo
        self.model_filename = model_filename
        self.max_tokens = max_tokens

        # Initialize components
        self.index = None
        self.metadata = None
        self.llm = None
        self.bert_tokenizer = None
        self.bert_model = None

        # Load components
        self._load_vector_store()
        self._load_llm()
        self._load_bert_models()

    def _load_vector_store(self):
        """Load Faiss index and metadata."""
        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.metadata)} artworks")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise

    def _load_llm(self):
        """Load and initialize the LLM."""
        try:
            model_path = hf_hub_download(repo_id=self.model_repo, filename=self.model_filename)
            self.llm = GPT4All(model=model_path, max_tokens=self.max_tokens)

            # Test loading with timeout
            func_timeout(120, self.llm.invoke, args=("Test",))
            logger.info("LLM loaded successfully")
        except FunctionTimedOut:
            logger.error("LLM loading timed out")
            raise RuntimeError("LLM loading timed out")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise

    def _load_bert_models(self):
        """Load BERT models for query encoding."""
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
            self.bert_model = BertModel.from_pretrained('prajjwal1/bert-tiny')
            logger.info("BERT models loaded")
        except Exception as e:
            logger.error(f"Failed to load BERT models: {e}")
            raise

    def encode_query(self, text: str) -> np.ndarray:
        """
        Encode query text into embedding.

        Args:
            text: Query text

        Returns:
            Normalized embedding vector
        """
        inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        # Pad to 640-dim
        query_emb = np.concatenate([emb, np.zeros(512)])
        # Normalize
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm
        return query_emb

    def search_similar_artworks(self, query_emb: np.ndarray, k: int = 5) -> tuple:
        """
        Search for similar artworks using Faiss.

        Args:
            query_emb: Query embedding
            k: Number of results

        Returns:
            Tuple of (distances, indices)
        """
        query_emb = query_emb.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_emb, k)
        return distances, indices

    def build_context(self, indices: np.ndarray) -> str:
        """
        Build context string from retrieved metadata.

        Args:
            indices: Indices of retrieved artworks

        Returns:
            Context string
        """
        retrieved_meta = [self.metadata[idx] for idx in indices[0]]
        context = "\n".join([
            f"Artist: {m['artist']}, Title: {m['title']}, Year: {m['year']}, Source: {m['dataset_source']}"
            for m in retrieved_meta
        ])
        return context, retrieved_meta

    def generate_critique(self, query: str) -> Dict[str, Any]:
        """
        Generate art critique for the given query.

        Args:
            query: User query about art style

        Returns:
            Dictionary with critique result and metadata
        """
        try:
            # Encode query
            query_emb = self.encode_query(query)

            # Search
            distances, indices = self.search_similar_artworks(query_emb)

            # Build context
            context, retrieved_meta = self.build_context(indices)

            # Create prompt
            full_prompt = f"""
Given this artwork metadata and similar artworks:
{context}

{query}

Write a short critique focusing on emotion, style, and artist intention.
Output in JSON format with fields: title, artist, critique_text, emotion_summary
"""

            # Generate critique
            response = self.llm.invoke(full_prompt)

            return {
                "query": query,
                "critique": response,
                "retrieved_artworks": retrieved_meta,
                "distances": distances.tolist()
            }

        except Exception as e:
            logger.error(f"Error generating critique: {e}")
            raise


def create_art_critic_generator(index_path: str, metadata_path: str) -> ArtCriticGenerator:
    """
    Factory function to create ArtCriticGenerator instance.

    Args:
        index_path: Path to Faiss index
        metadata_path: Path to metadata JSON

    Returns:
        ArtCriticGenerator instance
    """
    return ArtCriticGenerator(index_path, metadata_path)


def generate_art_critique(generator: ArtCriticGenerator, query: str) -> Dict[str, Any]:
    """
    Kedro node function to generate art critique.

    Args:
        generator: ArtCriticGenerator instance
        query: User query

    Returns:
        Critique result dictionary
    """
    return generator.generate_critique(query)


def create_faiss_index_and_metadata(embedded_dataset: pd.DataFrame) -> tuple:
    """
    Create Faiss index and metadata from embedded dataset.

    Args:
        embedded_dataset: DataFrame with embeddings and metadata

    Returns:
        Tuple of (index, metadata_list)
    """
    import numpy as np
    import faiss

    # Assume the dataset has columns for metadata and 'embedding' as list or array
    # For simplicity, assume it's similar to the notebook

    # Extract embeddings
    embeddings = []
    metadata = []

    for _, row in embedded_dataset.iterrows():
        # Assume embedding is a list or string
        emb = row['embedding']
        if isinstance(emb, str):
            emb = json.loads(emb)
        embeddings.append(emb)
        meta = {
            'artist': row.get('artist', ''),
            'title': row.get('title', ''),
            'year': row.get('year', ''),
            'dataset_source': row.get('dataset_source', '')
        }
        metadata.append(meta)

    embeddings = np.array(embeddings).astype('float32')

    # Create Faiss index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine
    # Normalize
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    return index, metadata