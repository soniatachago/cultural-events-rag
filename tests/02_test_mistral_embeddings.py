import sys
from pathlib import Path

# Ajouter le dossier scripts au path
sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))

import config

from dotenv import load_dotenv
import os
from mistralai import Mistral

# -----------------------------
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# -----------------------------
load_dotenv()  # charge les variables depuis .env

API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    raise ValueError("❌ La clé API Mistral est introuvable. Vérifie ton fichier .env")

# -----------------------------
# INITIALISATION CLIENT MISTRAL
# -----------------------------
client = Mistral(api_key=API_KEY)

# -----------------------------
# TEST EMBEDDINGS
# -----------------------------
response = client.embeddings.create(
    model=config.MODEL_EMBED,  # modèle défini dans config.py
    inputs=["Bonjour", "Festival de jazz"]
)

# Affichage des 5 premières valeurs du premier embedding
print(response.data[0].embedding[:5])
