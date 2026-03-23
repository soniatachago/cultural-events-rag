"""
# STEP(2) --> 02_rag_chunk_embeddings_indexation.py

Pipeline de création d'embeddings pour les événements et indexation FAISS.
"""

from mistralai import Mistral
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pathlib import Path
import pandas as pd
import pickle
import numpy as np
import faiss
import config
from dotenv import load_dotenv
import os


# -----------------------------
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# -----------------------------
load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    raise ValueError("❌ La clé API Mistral est introuvable. Vérifie ton fichier .env")


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
    df = pd.read_csv(input_path)
    expected_cols = ["event_id","title", "long_description", "start_date", "city", "venue", "tags", "region"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = "N/A"
    df = df.dropna(subset=["long_description"])
    df["long_description"] = df["long_description"].str.replace("\n", " ")
    print(f"✅ {len(df)} événements chargés")
    return df

# -----------------------------
# DÉCOUPAGE EN CHUNKS
# -----------------------------
def chunk_documents(df: pd.DataFrame, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP) -> list:
    """
    Découpe chaque événement en chunks pour génération d'embeddings.

    Args:
        df (pd.DataFrame): DataFrame des événements.
        chunk_size (int): Taille maximale d'un chunk.
        chunk_overlap (int): Nombre de caractères partagés entre chunks.

    Returns:
        list: Liste de dictionnaires {chunk, metadata}.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=config.SEPARATORS,
        length_function=len
    )
    all_chunks = []
    for _, row in df.iterrows():
        text = f"{row['title']} {row['long_description']}"
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append({
                "event_id": row.get("event_id", "N/A"),
                "title": row.get("title", "N/A"),
                "start_date": row.get("start_date", "N/A"),
                "city": row.get("city", "N/A"),
                "region": row.get("region", "N/A"),
                "venue": row.get("venue", "N/A"),
                "tags": row.get("tags", "N/A"),
                "chunk": chunk
            })
    print(f"✅ Total chunks générés : {len(all_chunks)}")
    return all_chunks

# -----------------------------
# GÉNÉRATION DES EMBEDDINGS
# -----------------------------
def generate_embeddings(chunks: list) -> list:
    """
    Génère les embeddings Mistral pour chaque chunk.

    Args:
        chunks (list): Liste de chunks avec metadata.

    Returns:
        list: Liste de dictionnaires {"embedding", "metadata"}.
    """
    embeddings = []
    for i in range(0, len(chunks), config.BATCH_SIZE):
        batch = [c["chunk"] for c in chunks[i:i + config.BATCH_SIZE]]
        try:
            response = client.embeddings.create(
                model=config.MODEL_EMBED_NAME,
                inputs=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            for j, emb in enumerate(batch_embeddings):
                embeddings.append({
                    "embedding": np.array(emb, dtype='float32'),
                    "metadata": chunks[i + j]
                })
            print(f"Batch {i // config.BATCH_SIZE + 1} traité ({len(batch)} chunks)")
        except Exception as e:
            print(f"❌ Erreur batch {i // config.BATCH_SIZE + 1}: {e}")
    print(f"🎯 {len(embeddings)} embeddings générés pour {len(chunks)} chunks")
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"✅ Embeddings sauvegardés dans {output_path}")

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
    vectors = np.array([e["embedding"] for e in embeddings], dtype='float32')
    metadata = [e["metadata"] for e in embeddings]

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    print(f"✅ Index FAISS créé avec {index.ntotal} vecteurs")

    # Assurer que le répertoire existe
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
    # faiss.write_index(index, f"{index_dir}/index.faiss")
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Index FAISS et métadonnées sauvegardés dans {index_dir}")
    return index, metadata

# -----------------------------
# RECHERCHE SEMANTIQUE
# -----------------------------
def semantic_search(query: str, index, metadata: list, k: int = 3):
    """
    Recherche sémantique dans l'index FAISS à partir d'une requête.

    Args:
        query (str): Question ou mot-clé.
        index (faiss.Index): Index FAISS.
        metadata (list): Liste de metadata.
        k (int): Nombre de résultats.

    Returns:
        list: Résultats les plus proches.
    """
    query_vector = client.embeddings.create(
        model=config.MODEL_EMBED_NAME,
        inputs=[query]
    ).data[0].embedding
    query_vector = np.array(query_vector, dtype='float32').reshape(1, -1)

    D, I = index.search(query_vector, k)
    results = [metadata[i] for i in I[0]]
    return results

# -----------------------------
# PIPELINE PRINCIPAL
# -----------------------------
def main():
    df_events = load_event_dataset(config.CSV_INPUT_PATH)
    chunks = chunk_documents(df_events)
    embeddings = generate_embeddings(chunks)
    save_embeddings(embeddings, config.EMBEDDINGS_OUTPUT)

    print("📂 Construction de l'index FAISS à partir des embeddings existants...")
    index, metadata = build_faiss_index_from_embeddings(embeddings, config.FAISS_INDEX_DIR)

    # query = "Visite sensorielle"
    # results = semantic_search(query, index, metadata, k=3)
    # print(f"\n🔎 Résultats de la recherche : {query}")
    # for r in results:
        # print(f"- {r['title']} ({r['start_date']}) - {r['city']}")

if __name__ == "__main__":
    main()
