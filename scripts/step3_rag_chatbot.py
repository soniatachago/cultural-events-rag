"""
# STEP(3) --> step3_rag_chatbot.py

Chatbot RAG pour recommandation d'événements culturels
- Utilise LangChain + FAISS + Mistral
- Permet d'interroger l'index FAISS pour retrouver des chunks pertinents
- Génère une réponse avec Mistral en utilisant un prompt RAG
"""

import faiss
import pickle
import numpy as np
from mistralai import Mistral
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from scripts.config import LOGGER, MAX_RETRIES, MODEL_EMBED_NAME, MODEL_LLM, FAISS_INDEX_PATH , METADATA_PATH, TOP_K
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
# CHARGEMENT INDEX FAISS
# -----------------------------
def load_faiss_index():
    """
    Charge l'index FAISS et les métadonnées associées.

    Returns:
        index (faiss.Index): index FAISS chargé
        metadata (list): liste de dictionnaires contenant les chunks et infos
    """
    
    try:
        # Conversion Path -> str pour compatibilité FAISS
        index = faiss.read_index(str(FAISS_INDEX_PATH))

        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)

        LOGGER.info(f"✅ Index FAISS chargé ({index.ntotal} vecteurs)")
        return index, metadata
    
    except Exception as e:
        LOGGER.error(f"❌ Erreur chargement FAISS: {e}")
        return None, None



# -----------------------------
# RECHERCHE VECTORIELLE (avec retry)
# -----------------------------
def retrieve_documents(query, index, metadata, k=TOP_K):
    """
    Récupère les k documents les plus proches d'une requête.

    Args:
        query (str): question ou texte de l'utilisateur
        index (faiss.Index): index FAISS
        metadata (list): liste des métadonnées correspondant aux vecteurs
        k (int): nombre de documents à retourner

    Returns:
        list: liste de chunks/métadonnées pertinents
    """
    
    for attempt in range(MAX_RETRIES):
        try:
            query_embedding = client.embeddings.create(
                model=MODEL_EMBED_NAME,
                inputs=[query]
            ).data[0].embedding

            query_vector = np.array(query_embedding, dtype="float32").reshape(1, -1)

            distances, indices = index.search(query_vector, k)

            docs = [metadata[i] for i in indices[0] if i < len(metadata)]

            LOGGER.info(f"Top {len(docs)} documents récupérés")
            return docs

        except Exception as e:
            LOGGER.warning(f"⚠️ Erreur embedding (tentative {attempt+1}) : {e}")
            time.sleep(2)

    LOGGER.error("❌ Échec récupération documents")
    return []



# -----------------------------
# DEDUPLICATION DES CHUNKS
# -----------------------------
def deduplicate_documents(docs):
    """
    Supprime les doublons d'événements en se basant sur event_id.
    Garde le premier chunk rencontré (le plus pertinent).
    """
    
    if not docs:
        LOGGER.warning("⚠️ Aucun documents à traiter pour la déduplication")
        return ""
        
    seen = set()
    unique_docs = []
    
    for doc in docs:
        event_id = doc.get("event_id")

        if event_id not in seen:
            seen.add(event_id)
            unique_docs.append(doc)

    LOGGER.info(f"{len(unique_docs)} documents après déduplication")
    return unique_docs
    

# -----------------------------
# CONSTRUCTION DU CONTEXTE
# -----------------------------
def build_context(docs):
    """
    Construit un contexte textuel à partir des documents récupérés.

    Args:
        docs (list): liste de dictionnaires représentant les chunks et metadata

    Returns:
        str: texte concaténé des documents pour le prompt RAG
    """
    
    if not docs:
        LOGGER.warning("⚠️ Aucun documents à traiter dapourns la construction du contexte")
        return ""
    
    context = ""
    
    for doc in docs:
        context += f"""
        Titre : {doc['title']}
        Date : {doc['start_date']}
        Ville : {doc['city']}
        Lieu : {doc['venue']}
        Description : {doc['chunk']}
    """
    
    return context


# -----------------------------
# PROMPT TEMPLATE
# -----------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Tu es un assistant spécialisé dans les événements culturels.

En utilisant uniquement les informations ci-dessous,
recommande des événements pertinents à l'utilisateur.

Événements disponibles :

{context}

Question utilisateur :
{question}

Réponds de manière claire et naturelle en recommandant les événements les plus pertinents.
"""
)


# -----------------------------
# GÉNÉRATION DE LA RÉPONSE
# -----------------------------
def generate_answer(question, context):
    """
    Génère une réponse de type chatbot à partir d'une question et d'un contexte.

    Args:
        question (str): question utilisateur
        context (str): contexte textuel issu des chunks FAISS

    Returns:
        str: réponse générée par Mistral
    """
    
    if not context:
        return "Aucun événement pertinent trouvé."
        
    prompt = prompt_template.format(context=context, question=question)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.complete(
                model=MODEL_LLM,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.choices[0].message.content

        except Exception as e:
            LOGGER.warning(f"⚠️ Erreur LLM (tentative {attempt+1}) : {e}")
            time.sleep(2)

    LOGGER.error("❌ Échec génération réponse")
    
    return "Une erreur est survenue lors de la génération de la réponse."


# -----------------------------
# CHATBOT RAG
# -----------------------------
def chatbot(question, index, metadata):
    """
    Pipeline complet de récupération et génération de réponse RAG.

    Args:
        question (str): question utilisateur
        index (faiss.Index): index FAISS
        metadata (list): métadonnées des vecteurs

    Returns:
        str: réponse finale du chatbot
    """
    
    if index is None or metadata is None:
        return "Le système n'est pas disponible (index non chargé)."

    
    try:
        docs = retrieve_documents(question, index, metadata)

        docs = deduplicate_documents(docs)

        if not docs:
            return "Aucun événement pertinent trouvé."

        context = build_context(docs)

        return generate_answer(question, context)

    except Exception as e:
        LOGGER.error(f"❌ Erreur pipeline chatbot : {e}")
        return "Une erreur interne pipeline RAG est survenue."


# -----------------------------
# MAIN
# -----------------------------
def main():
    """
    Lancement interactif du chatbot.
    """
        
    index, metadata = load_faiss_index()

    if index is None:
        LOGGER.error("Impossible de démarrer le chatbot")
        return

    print("\n🤖 Chatbot prêt ! (type 'quit' pour quitter)\n")

    while True:
        try:
            question = input("Utilisateur : ")

            if question.lower() == "quit":
                break

            answer = chatbot(question, index, metadata)

            print("\nBot :", answer, "\n")

        except KeyboardInterrupt:
            print("\nArrêt utilisateur")
            break

        except Exception as e:
            LOGGER.error(f"Erreur interaction : {e}")
            print("Une erreur est survenue. Réessaie.")


if __name__ == "__main__":
    main()
