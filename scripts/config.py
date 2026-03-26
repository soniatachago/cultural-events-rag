# config.py
"""
Configuration des chemins et paramètres pour le pipeline RAG.
"""

from pathlib import Path
import os
from datetime import datetime, timedelta

from scripts.logger import setup_logger
import time


# -----------------------------
# CONFIG LOGGING AVEC ROTATION
# -----------------------------
LOGGER = setup_logger()


# -----------------------------
# CONFIG RETRY
# -----------------------------
MAX_RETRIES = 3
RETRY_DELAY = 2
TIMEOUT = 10  # secondes


# -----------------------------
# API OpenAgenda
# -----------------------------
API_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"


# Date exacte d'il y a 1 an (365 jours)
one_year_ago = datetime.today() - timedelta(days=365)
one_year_ago_str = one_year_ago.strftime('%Y-%m-%d')

EXPECTED_CITY = "Marseille"

# print("Date il y a 1 an : ", one_year_ago_str)
	
# Paramètres de requête pour OpenAgenda
PARAMS = {
    "select": "uid, title_fr, description_fr, longdescription_fr, location_city, location_region, firstdate_begin, firstdate_end, location_tags",
    "where": f"longdescription_fr IS NOT NULL AND location_city = '{EXPECTED_CITY}' AND firstdate_begin >= '{one_year_ago_str}'",
    "order_by": "firstdate_begin"
    ,"limit": 100
}

# -----------------------------
# Chemins de fichiers pour sauvegarde des évènements
# -----------------------------
OUTPUT_PATH = Path(os.getenv("EVENTS_CSV_PATH", "../data/events_clean.csv"))

# Dossier pour sauvegarde optionnelle Markdown
MD_OUTPUT_DIR = Path(os.getenv("MD_OUTPUT_DIR", "../data/docs_md"))


# STEP(2) --> 02_pipeline_rag_chunk_embeddings_indexation.py

# -----------------------------
# Chemins fichiers/dossiers embeddings
# -----------------------------
CSV_INPUT_PATH = Path(os.getenv("CSV_INPUT_PATH", "../data/events_clean.csv"))
EMBEDDINGS_OUTPUT = Path(os.getenv("EMBEDDINGS_OUTPUT", "../vector_store/events_embeddings.pkl"))
FAISS_INDEX_PATH = Path(os.getenv("EMBEDDINGS_OUTPUT", "../vector_store/faiss_index"))

# -----------------------------
# Clés API et modèles Mistral
# -----------------------------
API_KEY = os.getenv("MISTRAL_API_KEY", "your_api_key")
MODEL_EMBED_NAME = "mistral-embed"			# modèle pour embeddings
MODEL_LLM = "mistral-small-latest"			# modèle pour génération de texte

# -----------------------------
# Paramètres chunking
# -----------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", " ", ""]

# -----------------------------
# Paramètres batch pour embeddings
# -----------------------------
BATCH_SIZE = 20


# STEP(3) --> 03_rag_chatbot_langchain_LLM_Mistral.py

# -----------------------------
# Chemins FAISS et métadonnées
# -----------------------------
FAISS_INDEX_DIR = Path("../vector_store/faiss_index")

FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
METADATA_PATH = FAISS_INDEX_DIR / "metadata.pkl"


# -----------------------------
# Paramètres recherche vectorielle
# -----------------------------
TOP_K = 3  # nombre de documents les plus proches à récupérer

# -----------------------------
# CONFIG EVALUATION RAG CHATBOT
# -----------------------------
# -----------------------------
# CONFIGURATION
# -----------------------------
DATASET_PATH = "evaluation/qa_dataset.json"  # ton fichier JSON de questions
RESULTS_PATH = "evaluation/rag_eval_results.json"
