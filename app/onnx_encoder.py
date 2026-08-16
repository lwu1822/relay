"""ONNX Runtime encoder — torch-free embedding inference.

Loads a pre-exported ONNX model + tokenizer and produces sentence
embeddings identical to SentenceTransformer('all-MiniLM-L6-v2') with
model_backend='onnx', but without requiring torch at runtime.

ONNX Runtime uses ~50MB RAM vs PyTorch's ~300-500MB, enabling deployment
on memory-constrained free tiers (e.g. Koyeb's 512MB free instance).

The ONNX model and tokenizer files are pre-baked into the Docker image
at build time (see Dockerfile builder stage) to avoid runtime downloads.
Mean pooling (replacing sentence-transformers' Pooling module) and the
encoder interface (callable with .encode(text) -> np.ndarray) are
implemented here so the rest of the cache logic is unchanged.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class OnnxEncoder:
    """Embedding encoder using ONNX Runtime + the tokenizers library.

    Produces the same embeddings as SentenceTransformer('all-MiniLM-L6-v2')
    with model_backend='onnx', but without requiring torch at runtime.
    """

    def __init__(self, model_dir: str, max_length: int = 256):
        # Load the fast tokenizer from the pre-saved tokenizer.json file.
        self.tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        self.tokenizer.enable_truncation(max_length=max_length)
        self.max_length = max_length

        # Load the ONNX model with CPU execution provider.
        self.session = ort.InferenceSession(
            f"{model_dir}/model.onnx",
            providers=["CPUExecutionProvider"],
        )

        # Determine which inputs the ONNX model expects. BERT-based models
        # typically take input_ids, attention_mask, and token_type_ids.
        self._input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text: str) -> np.ndarray:
        """Encode text to a 1-D numpy array (not L2-normalized).

        L2 normalization is done by SemanticCache._encode after this
        method returns, matching the existing interface.
        """
        encoding = self.tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # Some BERT-based models also expect token_type_ids (all zeros for
        # single-sentence encoding).
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, feeds)
        last_hidden_state = outputs[0]  # (1, seq_len, hidden_dim)

        # Mean pooling with attention mask — replaces sentence-transformers'
        # Pooling module. Each token's embedding is masked, summed, and
        # divided by the number of non-padding tokens.
        mask = attention_mask[:, :, np.newaxis]  # (1, seq_len, 1)
        masked = last_hidden_state * mask
        summed = masked.sum(axis=1)  # (1, hidden_dim)
        counts = np.clip(mask.sum(axis=1), 1e-9, float("inf"))  # (1, 1)
        pooled = summed / counts  # (1, hidden_dim)

        return pooled[0]  # (hidden_dim,)
