# tests/test_rag_chatbot.py

import pytest
from scripts import step3_rag_chatbot as event_cb
import numpy as np


def test_build_context():
    docs = [
        {
            "title": "Concert Jazz",
            "start_date": "2026-03-01",
            "city": "Paris",
            "venue": "Salle X",
            "chunk": "Un super concert"
        }
    ]

    context = event_cb.build_context(docs)

    assert "Concert Jazz" in context
    assert "Paris" in context
    assert "Un super concert" in context


def test_generate_answer(monkeypatch):

    class MockResponse:
        class Choice:
            class Message:
                content = "Réponse simulée"
            message = Message()
        choices = [Choice()]

    class MockClient:
        class chat:
            @staticmethod
            def complete(model, messages):
                return MockResponse()

    # Remplacer client global dans ton script
    monkeypatch.setattr(event_cb, "client", MockClient())

    result = event_cb.generate_answer("Question", "Contexte")

    assert "Réponse simulée" in result


def test_retrieve_documents(monkeypatch):

    # Mock embedding
    class MockEmbeddingResponse:
        class DataItem:
            embedding = [0.1] * 5
        data = [DataItem()]

    class MockClient:
        class embeddings:
            @staticmethod
            def create(model, inputs):
                return MockEmbeddingResponse()

    monkeypatch.setattr(event_cb, "client", MockClient())

    # Mock FAISS index
    class MockIndex:
        def search(self, vector, k):
            return None, [[0]]

    metadata = [
        {"title": "Event test", "chunk": "desc"}
    ]

    docs = event_cb.retrieve_documents("concert", MockIndex(), metadata, k=1)

    assert len(docs) == 1
    assert docs[0]["title"] == "Event test"
	

def test_chatbot_pipeline(monkeypatch):

    # Mock retrieve_documents
    def mock_retrieve(query, index, metadata):
        return [{
            "event_id": "1",
            "title": "Concert",
            "start_date": "2026",
            "city": "Paris",
            "venue": "Salle X",
            "chunk": "desc"
        }]

    # Mock generate_answer
    def mock_generate(question, context):
        return "Réponse finale"

    # Mock index + metadata (non None)
    fake_index = object()
    fake_metadata = [{}]

    monkeypatch.setattr(event_cb, "retrieve_documents", mock_retrieve)
    monkeypatch.setattr(event_cb, "generate_answer", mock_generate)

    result = event_cb.chatbot("Question ?", fake_index, fake_metadata)

    assert result == "Réponse finale"

