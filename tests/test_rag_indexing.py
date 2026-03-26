import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from scripts.step2_rag_indexing import (
    load_event_dataset,
    chunk_documents,
    generate_embeddings,
    build_faiss_index_from_embeddings
)

# -----------------------------
# MOCK DATA
# -----------------------------
MOCK_DF = pd.DataFrame({
    "event_id": ["1"],
    "title": ["Festival Jazz"],
    "long_description": ["Un grand festival de jazz avec plusieurs artistes."],
    "start_date": ["2025-06-01"],
    "city": ["Marseille"],
    "region": ["PACA"],
    "venue": ["Parc"],
    "tags": ["musique"]
})

# -----------------------------
# TEST 1 — Chargement dataset
# -----------------------------
def test_load_event_dataset(tmp_path):
    file_path = tmp_path / "test.csv"
    MOCK_DF.to_csv(file_path, index=False)

    df = load_event_dataset(file_path)

    assert not df.empty
    assert "long_description" in df.columns


# -----------------------------
# TEST 2 — Chunking
# -----------------------------
def test_chunk_documents():
    chunks = chunk_documents(MOCK_DF, chunk_size=50, chunk_overlap=10)

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    chunk = chunks[0]
    assert "chunk" in chunk
    assert "title" in chunk
    assert "event_id" in chunk


# -----------------------------
# TEST 3 — Embeddings (mock Mistral)
# -----------------------------
@patch("scripts.step2_rag_indexing.client")
def test_generate_embeddings(mock_client):
    # Mock réponse API Mistral
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3])
    ]

    mock_client.embeddings.create.return_value = mock_response

    chunks = [{
        "chunk": "Test texte",
        "event_id": "1",
        "title": "Test"
    }]

    embeddings = generate_embeddings(chunks)

    assert len(embeddings) == 1
    assert "embedding" in embeddings[0]
    assert isinstance(embeddings[0]["embedding"], np.ndarray)


# -----------------------------
# TEST 4 — FAISS index
# -----------------------------
def test_build_faiss_index(tmp_path):
    embeddings = [
        {
            "embedding": np.array([0.1, 0.2, 0.3], dtype="float32"),
            "metadata": {"title": "Event 1"}
        },
        {
            "embedding": np.array([0.2, 0.1, 0.4], dtype="float32"),
            "metadata": {"title": "Event 2"}
        }
    ]

    index_dir = tmp_path / "faiss_index"

    index, metadata = build_faiss_index_from_embeddings(embeddings, index_dir)

    assert index.ntotal == 2
    assert len(metadata) == 2

    # Vérifier fichiers créés
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "metadata.pkl").exists()
