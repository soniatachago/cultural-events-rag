from scripts.step3_rag_chatbot import deduplicate_documents

def test_deduplicate_documents():

    docs = [
        {"event_id": "1", "title": "Event A"},
        {"event_id": "1", "title": "Event A chunk 2"},
        {"event_id": "2", "title": "Event B"},
        {"event_id": "3", "title": "Event C"},
        {"event_id": "2", "title": "Event B chunk 2"}
    ]

    unique_docs = deduplicate_documents(docs)

    assert len(unique_docs) == 3

    event_ids = [d["event_id"] for d in unique_docs]

    assert len(set(event_ids)) == len(unique_docs)
