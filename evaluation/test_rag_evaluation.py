"""
Script d'évaluation automatique du RAG (step3_rag_chatbot.py)

Objectif :
- Évaluer la qualité des réponses RAG pour un dataset métier de questions.
- Vérifier pertinence géographique, mots-clés attendus, et cas négatifs.
- Générer un JSON de résultats pour analyse ultérieure.

Usage :
    python scripts/eval_rag.py
"""

import json
import os
import re
from scripts.step3_rag_chatbot import load_faiss_index, chatbot

from scripts.config import LOGGER, DATASET_PATH, RESULTS_PATH


# -----------------------------
# CHARGEMENT DU DATASET
# -----------------------------
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

LOGGER.info(f"✅ Dataset chargé : {len(dataset)} questions")

# -----------------------------
# CHARGEMENT INDEX FAISS
# -----------------------------
index, metadata = load_faiss_index()
if index is None or metadata is None:
    raise RuntimeError("Impossible de charger FAISS. Vérifie la config.")


# -----------------------------
# FONCTION D'ÉVALUATION
# -----------------------------
def evaluate_rag_question(q):
    """
    Évalue une question RAG et compare avec le dataset.

    Args:
        q (dict): question du dataset (avec expected_city, keywords_expected, expected_result...)

    Returns:
        dict: résultats incluant question, réponse RAG, statut réussite/cas négatif
    """
    question_text = q["question"]

    # Appel RAG
    response = chatbot(question_text, index, metadata)

    # Détection cas négatif
    is_expected_none = q.get("expected_result") == "none"
    response_none = response.lower() in ["aucun événement pertinent trouvé.", "none", ""]

    # Vérification pertinence géographique
    city_ok = True
    if "expected_city" in q and not is_expected_none:
        city_ok = q["expected_city"].lower() in response.lower()

    # Vérification présence mots-clés
    keywords_ok = True
    if "keywords_expected" in q and not is_expected_none:
        # Au moins un mot-clé attendu doit apparaître dans la réponse
        keywords_ok = any(
            re.search(rf'\b{re.escape(kw)}\b', response, re.IGNORECASE)
            for kw in q["keywords_expected"]
        )

    # Statut global
    success = False
    if is_expected_none:
        success = response_none
    else:
        success = city_ok and keywords_ok

    return {
        "id": q.get("id"),
        "question": question_text,
        "expected_city": q.get("expected_city"),
        "keywords_expected": q.get("keywords_expected"),
        "expected_result": q.get("expected_result", "some"),
        "rag_response": response,
        "success": success,
        "city_ok": city_ok,
        "keywords_ok": keywords_ok
    }


# -----------------------------
# ÉVALUATION TOTALE
# -----------------------------
results = []

for q in dataset:
    res = evaluate_rag_question(q)
    results.append(res)
    LOGGER.info(f"Question ID {q.get('id')} -> Success: {res['success']}")

# -----------------------------
# SAUVEGARDE DES RÉSULTATS
# -----------------------------
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

LOGGER.info(f"✅ Évaluation terminée. Résultats sauvegardés dans {RESULTS_PATH}")

# -----------------------------
# MÉTRIQUES SIMPLES
# -----------------------------
total = len(results)
success_count = sum(r["success"] for r in results)
accuracy = success_count / total * 100

LOGGER.info(f"📊 Total questions : {total}")
LOGGER.info(f"📊 Réponses correctes : {success_count}")
LOGGER.info(f"📊 Accuracy globale : {accuracy:.2f}%")
