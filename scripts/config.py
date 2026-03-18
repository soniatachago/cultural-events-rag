# config.py
"""
Configuration des chemins et paramètres pour le pipeline RAG.
"""

from pathlib import Path
import os


# STEP(1) --> 01_collect_events.py

# -----------------------------
# API OpenAgenda
# -----------------------------
API_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

# Paramètres de requête pour OpenAgenda
PARAMS = {
    "select": "uid, title_fr, description_fr, longdescription_fr, location_city, location_region, firstdate_begin, firstdate_end, location_tags",
    "where": "description_fr IS NOT NULL AND location_city = 'Marseille' AND firstdate_begin >= '2025-01-01'",
    "order_by": "firstdate_begin",
    "limit": 100
}

# -----------------------------
# Chemins de fichiers pour sauvegarde des évènements
# -----------------------------
OUTPUT_PATH = Path(os.getenv("EVENTS_CSV_PATH", "../data/events_clean.csv"))

# Dossier pour sauvegarde optionnelle Markdown
MD_OUTPUT_DIR = Path(os.getenv("MD_OUTPUT_DIR", "../data/docs_md"))


