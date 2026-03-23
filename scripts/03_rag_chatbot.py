"""
# STEP(3) --> 03_rag_chatbot_langchain_LLM_Mistral.py

Chatbot RAG pour recommandation d'événements culturels
- Utilise LangChain + FAISS + Mistral
- Permet d'interroger l'index FAISS pour retrouver des chunks pertinents
- Génère une réponse avec Mistral en utilisant un prompt RAG
"""

import faiss
import pickle
import numpy as np
from mistralai import Mistral
from langchain.prompts import PromptTemplate
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
# CHARGEMENT INDEX FAISS
# -----------------------------
def load_faiss_index():
    """
    Charge l'index FAISS et les métadonnées associées.

    Returns:
        index (faiss.Index): index FAISS chargé
        metadata (list): liste de dictionnaires contenant les chunks et infos
    """
    # Conversion Path -> str pour compatibilité FAISS
    index = faiss.read_index(str(config.FAISS_INDEX_PATH))

    with open(config.METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    print(f"✅ Index FAISS chargé ({index.ntotal} vecteurs)")

    return index, metadata



# -----------------------------
# RECHERCHE VECTORIELLE
# -----------------------------
def retrieve_documents(query, index, metadata, k=config.TOP_K):
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
    query_embedding = client.embeddings.create(
        model=config.MODEL_EMBED,
        inputs=[query]
    ).data[0].embedding

    query_vector = np.array(query_embedding, dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    docs = [metadata[i] for i in indices[0]]

    return docs


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
    prompt = prompt_template.format(context=context, question=question)

    response = client.chat.complete(
        model=config.MODEL_LLM,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


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
    docs = retrieve_documents(question, index, metadata)
    context = build_context(docs)
    answer = generate_answer(question, context)
    return answer


# -----------------------------
# MAIN
# -----------------------------
def main():
    """
    Lancement interactif du chatbot.
    """
    index, metadata = load_faiss_index()

    print("\n🤖 Chatbot événements culturels prêt !")
    print("Tape 'quit' pour quitter.\n")

    while True:
        question = input("Utilisateur : ")
        if question.lower() == "quit":
            break
        answer = chatbot(question, index, metadata)
        print("\nBot :", answer, "\n")


if __name__ == "__main__":
    main()
