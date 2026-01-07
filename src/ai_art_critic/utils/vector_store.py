# Vector Store Module for AI Art Critic
# Handles Faiss index operations for art retrieval

import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any
from transformers import BertTokenizer, BertModel
import torch

class ArtVectorStore:
    def __init__(self, index_path: str = "art_embeddings.index", metadata_path: str = "art_metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = None
        self.bert_tokenizer = None
        self.bert_model = None
        self._load_store()

    def _load_store(self):
        """Load Faiss index and metadata from disk."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"Loaded Faiss index with {self.index.ntotal} vectors")
        else:
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata for {len(self.metadata)} artworks")
        else:
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        # Load BERT for text encoding
        self.bert_tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
        self.bert_model = BertModel.from_pretrained('prajjwal1/bert-tiny')

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text query to embedding."""
        inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def query_text(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the store with text and return top-k similar artworks."""
        if self.index is None:
            raise ValueError("Index not loaded")

        # Encode query
        text_emb = self._encode_text(query_text)
        query_emb = np.concatenate([text_emb, np.zeros(512)])  # Pad for image

        # Normalize
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        # Search
        query_emb = query_emb.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_emb, k=top_k)

        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'distance': float(distances[0][i]),
                'metadata': self.metadata[idx]
            })
        return results

    def add_artwork(self, text_emb: np.ndarray, image_emb: np.ndarray, metadata: Dict[str, Any]):
        """Add a new artwork to the store (in-memory, save separately)."""
        if text_emb.size == 0 and image_emb.size == 0:
            raise ValueError("At least one embedding must be provided")

        # Combine embeddings
        if text_emb.size > 0 and image_emb.size > 0:
            combined = np.concatenate([text_emb, image_emb])
        elif text_emb.size > 0:
            combined = np.concatenate([text_emb, np.zeros(512)])
        else:
            combined = np.concatenate([np.zeros(128), image_emb])

        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        # Add to index
        self.index.add(combined.astype('float32').reshape(1, -1))
        self.metadata.append(metadata)

    def save_store(self):
        """Save the updated index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)
        print("Vector store saved")