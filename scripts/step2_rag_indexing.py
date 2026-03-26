"""
# STEP(2) --> step2_rag_indexing.py

Pipeline de création d'embeddings pour les événements et indexation FAISS.
"""

from mistralai import Mistral
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from pathlib import Path
import pandas as pd
import pickle
import numpy as np
import faiss
from scripts.config import LOGGER, MAX_RETRIES, RETRY_DELAY, CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS, MODEL_EMBED_NAME, BATCH_SIZE, EMBEDDINGS_OUTPUT, CSV_INPUT_PATH, FAISS_INDEX_DIR
from dotenv import load_dotenv
import os


# -----------------------------
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# -----------------------------
load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    raise ValueError("❌ Clé API Mistral est manquante. Vérifier le fichier .env")


# -----------------------------
# INITIALISATION CLIENT MISTRAL
# -----------------------------
client = Mistral(api_key=API_KEY)

# -----------------------------
# CHARGEMENT DES ÉVÉNEMENTS
# -----------------------------
def load_event_dataset(input_path: Path) -> pd.DataFrame:
    """
    Charge le dataset des événements depuis un CSV.

    Args:
        input_path (Path): Chemin vers le CSV.

    Returns:
        pd.DataFrame: DataFrame nettoyée avec les colonnes attendues.
    """
    
    try:
        df = pd.read_csv(input_path)
        expected_cols = ["event_id","title", "long_description", "start_date", "city", "venue", "tags", "region"]
        
        for col in expected_cols:
            if col not in df.columns:
                df[col] = "N/A"
                
        df = df.dropna(subset=["long_description"])
        df["long_description"] = df["long_description"].str.replace("\n", " ")
        
        LOGGER.info(f"✅ {len(df)} événements chargés")
        return df
    
    except Exception as e:
        LOGGER.error(f"❌ Erreur chargement dataset: {e}")
        return pd.DataFrame()    


# -----------------------------
# DÉCOUPAGE EN CHUNKS
# -----------------------------
def chunk_documents(df: pd.DataFrame, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """
    Découpe chaque événement en chunks pour génération d'embeddings.

    Args:
        df (pd.DataFrame): DataFrame des événements.
        chunk_size (int): Taille maximale d'un chunk.
        chunk_overlap (int): Nombre de caractères partagés entre chunks.

    Returns:
        list: Liste de dictionnaires {chunk, metadata}.
    """
    
    if df.empty:
        LOGGER.warning("⚠️ DataFrame vide - aucun chunk")
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len
    )
    
    all_chunks = []
    
    for _, row in df.iterrows():
        try:
            text = f"{row['title']} {row['long_description']}"
            chunks = splitter.split_text(text)
            
            for chunk in chunks:
                all_chunks.append({
                    "event_id": row.get("event_id", "N/A"),
                    "title": row.get("title", "N/A"),
                    "start_date": row.get("start_date", "N/A"),
                    "end_date": row.get("start_date", "N/A"),
                    "city": row.get("city", "N/A"),
                    "region": row.get("region", "N/A"),
                    "venue": row.get("venue", "N/A"),
                    "tags": row.get("tags", "N/A"),
                    "chunk": chunk
                })
                
        except Exception as e:
            LOGGER.warning(f"⚠️ Erreur chunk event {row.get('event_id')}: {e}")

    LOGGER.info(f"✅ {len(all_chunks)} chunks générés")
    return all_chunks
    

# -----------------------------
# GÉNÉRATION DES EMBEDDINGS AVEC RETRY
# -----------------------------
def generate_embeddings(chunks: list) -> list:
    """
    Génère les embeddings Mistral pour chaque chunk.

    Args:
        chunks (list): Liste de chunks avec metadata.

    Returns:
        list: Liste de dictionnaires {"embedding", "metadata"}.
    """
    
    if not chunks:
        LOGGER.warning("⚠️ Aucun chunk à vectoriser")
        return []
        
    embeddings = []
    
    for i in range(0, len(chunks), BATCH_SIZE):

        batch = [c["chunk"] for c in chunks[i:i + BATCH_SIZE]]

        for attempt in range(1, MAX_RETRIES + 1):

            try:
                response = client.embeddings.create(
                    model=MODEL_EMBED_NAME,
                    inputs=batch
                )

                batch_embeddings = [item.embedding for item in response.data]

                for j, emb in enumerate(batch_embeddings):
                    embeddings.append({
                        "embedding": np.array(emb, dtype='float32'),
                        "metadata": chunks[i + j]
                    })

                LOGGER.info(f"✅ Batch {i // BATCH_SIZE + 1} OK ({len(batch)})")
                break

            except Exception as e:
                LOGGER.warning(f"⚠️ Erreur batch {i // BATCH_SIZE + 1} - tentative {attempt}: {e}")

                if attempt == MAX_RETRIES:
                    LOGGER.error("❌ Batch définitivement échoué")
                else:
                    time.sleep(RETRY_DELAY * attempt)

    LOGGER.info(f"🎯 {len(embeddings)} embeddings générés")
    return embeddings
    

# -----------------------------
# SAUVEGARDE DES EMBEDDINGS
# -----------------------------
def save_embeddings(embeddings: list, output_path: Path):
    """
    Sauvegarde les embeddings dans un fichier pickle.

    Args:
        embeddings (list): Liste des embeddings et metadata.
        output_path (Path): Chemin du fichier pickle.
    """
    
    if not embeddings:
        LOGGER.error("❌ Aucun embedding à sauvegarder")
        return
    
    try:    
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(embeddings, f)
        
        LOGGER.info(f"💾 Embeddings sauvegardés : {output_path}")

    except Exception as e:
        LOGGER.error(f"❌ Erreur sauvegarde embeddings: {e}")


# -----------------------------
# CONSTRUCTION DE L'INDEX FAISS
# -----------------------------
def build_faiss_index_from_embeddings(embeddings: list, index_dir: Path):
    """
    Crée un index FAISS et sauvegarde les vecteurs et métadonnées.

    Args:
        embeddings (list): Liste d'embeddings avec metadata.
        index_dir (Path): Répertoire de sauvegarde de l'index FAISS et métadonnées.
    
    Returns:
        index, metadata
    """
    
    if not embeddings:
        LOGGER.error("❌ Aucun embedding pour FAISS")
        return None, None
        
    try:
        vectors = np.array([e["embedding"] for e in embeddings], dtype='float32')
        metadata = [e["metadata"] for e in embeddings]

        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
    
        LOGGER.info(f"✅ FAISS index créé ({index.ntotal} vecteurs)")

        # Vérifier que le répertoire existe
        index_dir.mkdir(parents=True, exist_ok=True)

        index_file = index_dir / "index.faiss"
        metadata_file = index_dir / "metadata.pkl"
        
        # Supprimer les anciens fichiers si présents
        if index_file.exists():
            index_file.unlink()   # supprime index.faiss
        if metadata_file.exists():
            metadata_file.unlink()  # supprime metadata.pkl
        
        # Sauvegarder
        faiss.write_index(index, str(index_file))
        
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)

        LOGGER.info("💾 Index FAISS sauvegardé")

        return index, metadata
        
    except Exception as e:
        LOGGER.error(f"❌ Erreur FAISS: {e}")
        return None, None


# -----------------------------
# PIPELINE PRINCIPAL
# -----------------------------
def main():
    
    LOGGER.info("🚀 Démarrage pipeline RAG indexing")
    
    df_events = load_event_dataset(CSV_INPUT_PATH)
    
    if df_events.empty:
        LOGGER.error("❌ Arrêt pipeline (dataset vide)")
        return
        
    chunks = chunk_documents(df_events)
    
    embeddings = generate_embeddings(chunks)
    
    if not embeddings:
        LOGGER.error("❌ Arrêt pipeline (embeddings vides)")
        return
    
    save_embeddings(embeddings, EMBEDDINGS_OUTPUT)

    LOGGER.info("📂 Construction index FAISS...")
    index, metadata = build_faiss_index_from_embeddings(embeddings, FAISS_INDEX_DIR)

    LOGGER.info("\n🎯 Pipeline terminé avec succès")


if __name__ == "__main__":
    main()
