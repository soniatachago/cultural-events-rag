from mistralai import Mistral
from pathlib import Path
import pandas as pd
import pickle
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
CSV_INPUT_PATH = "data/events_clean.csv"           # CSV des événements pré-traités
EMBEDDINGS_OUTPUT = "vector_store/events_embeddings.pkl"  # Fichier de sortie embeddings
MODEL_EMBED_NAME = "mistral-embed"                # Modèle Mistral pour embeddings
API_KEY = "VZNRc1Lf2t0ZDWhqvJLKAaDgdMfW9M8W"                         # Remplacer par ta clé API Mistral
BATCH_SIZE = 20                                   # Nombre de textes par batch

# -----------------------------
# INITIALISATION CLIENT MISTRAL
# -----------------------------
client = Mistral(api_key=API_KEY)

# -----------------------------
# CHARGEMENT DES DONNÉES
# -----------------------------
def load_event_dataset(input_path: str) -> pd.DataFrame:
    """
    Charge les événements depuis le CSV nettoyé.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {input_path}")

    df = pd.read_csv(input_path)

    # Vérification des colonnes nécessaires
    expected_cols = ["title", "long_description", "start_date", "city", "venue", "tags", "region"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = "N/A"  # Valeur par défaut si colonne manquante

    df = df.dropna(subset=["long_description"])
    df["long_description"] = df["long_description"].str.replace("\n", " ")
    
    print(f"✅ Dataset chargé : {len(df)} événements")
    return df

# -----------------------------
# PRÉPARATION DES DOCUMENTS
# -----------------------------
def prepare_documents(df: pd.DataFrame) -> list:
    """
    Construit les documents texte à vectoriser.
    """
    documents = []
    for _, row in df.iterrows():
        text = f"""
        Event_id: {row.get('event_id', 'N/A')}
        Type évènement : {row.get('tags', 'N/A')}
        Titre: {row.get('title', 'N/A')}
        Date: {row.get('start_date', 'N/A')}
        Ville: {row.get('city', 'N/A')}
        Region: {row.get('region', 'N/A')}
        Lieu: {row.get('venue', 'N/A')}
        Description: {row.get('long_description', 'N/A')}
        """
        documents.append(text.strip())
    return documents

# -----------------------------
# GÉNÉRATION DES EMBEDDINGS
# -----------------------------
def generate_embeddings_mistral(documents: list, client: Mistral, model: str = MODEL_EMBED_NAME, batch_size: int = BATCH_SIZE) -> list:
    """
    Génère les embeddings pour une liste de documents via Mistral en batch.
    """
    if not documents:
        print("⚠️ La liste de documents est vide. Aucun embedding généré.")
        return []

    embeddings = []
    # print(f"⏳ Génération des embeddings pour {len(documents)} documents...")

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        try:
            response = client.embeddings.create(model=model, inputs=batch)
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            print(f"✅ Batch {i // batch_size + 1} traité ({len(batch)} docs)")
        except Exception as e:
            print(f"❌ Erreur batch {i // batch_size + 1} : {e}")

    print(f"🎯 {len(embeddings)} embeddings générés pour {len(documents)} documents.")
    return embeddings

# -----------------------------
# SAUVEGARDE DES EMBEDDINGS
# -----------------------------
def save_embeddings(embeddings: list, events_df: pd.DataFrame, output_path: str):
    """
    Sauvegarde les embeddings et les événements dans un fichier .pkl
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "events": events_df.to_dict(orient="records")
        }, f)
    print(f"✅ Embeddings sauvegardés dans {output_path} (total {len(embeddings)} embeddings)")

# -----------------------------
# PIPELINE PRINCIPAL
# -----------------------------
def main():
    print("📥 Chargement du dataset d'événements...")
    df_events = load_event_dataset(CSV_INPUT_PATH)

    print("📝 Préparation des documents...")
    documents = prepare_documents(df_events)

    print(f"⏳ Génération des embeddings pour {len(documents)} documents...")
    embeddings = generate_embeddings_mistral(documents, client)

    print("💾 Sauvegarde des embeddings...")
    save_embeddings(embeddings, df_events, EMBEDDINGS_OUTPUT)

    print("✅ Pipeline terminé.")

if __name__ == "__main__":
    main()
