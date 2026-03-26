# Assistant intelligent RAG de recommandation d'Événements Culturels


---

## 📌 Présentation du projet

Ce projet met en œuvre un **chatbot intelligent basé sur le RAG (Retrieval-Augmented Generation)** appliqué aux événements culturels.

Il permet de fournir des **recommandations personnalisées** en combinant :

- Une base de données d’événements (CSV) exportée à partir de Open Agenda
- Une recherche sémantique avec FAISS
- Un modèle de langage (**Mistral LLM**) pour générer des réponses naturelles
- Une orchestration via **LangChain**

---

## 🎯 Objectifs

- Charger et nettoyer des données d’événements culturels récents de moins d’un an
- Découper les textes (chunking) pour faciliter le traitement
- Générer des embeddings avec Mistral
- Construire un index vectoriel performant avec FAISS
- Orchestrer un pipeline RAG avec LangChain
- Développer un chatbot interactif capable de fournir des recommandations personnalisées et augmentées par des réponses précises issues de données d’événements sur un périmètre géographique connu

---

## 🏗 Architecture du pipeline RAG

flowchart TD

    %% =========================
    %% DATA INGESTION
    %% =========================
    subgraph INGESTION
        A["🌐 OpenAgenda API"]
        B["📥 Collecte + nettoyage"]
        C["📄 events_clean.csv"]
        A --> B --> C
    end

    %% =========================
    %% INDEXATION
    %% =========================
    subgraph INDEXATION
        D["📄 Input: Dataset d'évènements CSV"]
        E["✂️ Chunking<br/>Découpage des descriptions"]
        F["🧠 Embeddings (Mistral)<br/>Vectorisation sémantique"]
        G["📊 Vecteurs + metadata"]
        H["📦 Index FAISS<br/>Stockage + recherche par similarité"]
        I["💾 index.faiss + metadata.pkl"]

        D --> E --> F --> G --> H --> I
    end

    %% =========================
    %% RAG CHATBOT
    %% =========================
    subgraph CHATBOT
        J["👤 Question utilisateur"]
        K["🧠 Embedding requête"]
        L["🔎 Recherche FAISS (Top K)documents similaires"]
        M["📚 Contexte enrichi"]
        N["📝 Prompt RAG<br/>Injection du contexte"]
        O["🤖 LLM Mistral<br/>Génération contrôlée"]
        P["💬 Réponse pertinente"]

        J --> K --> L --> M --> N --> O --> P
    end

    %% Lien entre index et chatbot
    I --> L

---

## ⚙️ Fonctionnement (How it works)

1. **Collecte & Prétraitement des données :**  
   Extraction d’événements culturels via Open Agenda, puis sauvegarde dans un fichier CSV structuré.

2. **Chargement des données :**  
   Lecture du fichier CSV contenant les événements.

3. **Nettoyage & normalisation :**  
   Suppression des évènements avec données critiques manquantes (ex: `long_description`) et homogénéisation du texte.

4. **Chunking (segmentation) :**  
   Découpage des descriptions en segments plus petits pour optimiser la vectorisation.

5. **Génération des embeddings :**  
   Transformation des chunks en vecteurs numériques via l’API Mistral.

6. **Indexation vectorielle (FAISS) :**  
   Stockage des embeddings dans un index FAISS pour une recherche rapide et efficace.

7. **Recherche sémantique :**  
   Identification des documents les plus pertinents à partir de la requête utilisateur.

8. **Génération de réponse (RAG) :**  
   Injection du contexte récupéré dans un prompt, puis génération de la réponse via le modèle Mistral.

9. **Interaction utilisateur :**  
   Le chatbot fournit des recommandations pertinentes en langage naturel, basées uniquement sur les données indexées.

---

## 🧠 Concepts clés en IA

### 🔹 `RAG (Retrieval-Augmented Generation)`
Combinaison de :
- Recherche d’informations pertinentes (retrieval)
- Génération de texte (LLM)

➡️ Permet d’améliorer la précision des réponses en s’appuyant sur des données réelles.

### 🔹 `Embeddings`
Représentation vectorielle des textes permettant de mesurer la similarité sémantique.

➡️ Deux textes proches auront des vecteurs proches.

### 🔹 `FAISS`
Bibliothèque optimisée pour la recherche rapide dans des bases de vecteurs.

➡️ Permet de retrouver instantanément les événements les plus pertinents.

---

## 💻 Installation & Exécution

### Pré-requis
- Python >= 3.11
- Git
- Connexion internet (pour API Mistral)
- OS: Windows

### 1. Cloner le projet
```bash
git clone https://github.com/soniatachago/cultural-events-rag.git
cd cultural-events-rag
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API Mistral

Créer un fichier `.env` et ajouter: `MISTRAL_API_KEY = your_api_key`.

### 5. Exécuter le pipeline depuis la racine du projet (`cultural-events-rag`)

```bash
# Étape 1: Collecte des événements culturels et création du dataset CSV
python -m scripts.step1_data_collection
# Sortie: data/events_clean.csv

# Étape 2: Transformation des textes, découpage en chunks, génération d'embeddings et indexation FAISS
python -m scripts.step2_rag_indexing
# Sortie: 
#   vector_store/events_embeddings.pkl
#   vector_store/faiss_index/index.faiss
#   vector_store/faiss_index/metadata.pkl

Étape 3: Lancer le chatbot RAG interactif
python -m scripts.step3_rag_chatbot

```

### 6. Exemple d'utilisation

```text
Utilisateur : Quels sont les événements culturels en plein air à Marseille ?

Bot : Voici les événements culturels en plein air à Marseille qui pourraient vous intéresser :

1. **Visite sensorielle : balade des sens à travers l’histoire de Marseille**
   📅 **Date** : 17 mai 2025 à 17h
   📍 **Lieu** : Non précisé (probablement en extérieur)
   🎯 **Pourquoi y aller ?** Une expérience immersive où vous explorerez l’histoire de Marseille à travers vos sens : épices, musiques, tissus et bien plus. Parfait pour une découverte originale et sensorielle de la ville.

2. **Fête du Vélo à Malpassé - Marseille**
   📅 **Date** : 30 avril 2025 de 14h à 17h
   📍 **Lieu** : Musée de la Moto, 18 traverse Saint Paul, 13013 Marseille
   🎯 **Pourquoi y aller ?** Un événement festif et sportif avec des surprises, organisé dans le cadre des Cités Éducatives. Idéal pour les amateurs de vélo et les familles !

Ces deux événements offrent une belle occasion de profiter de la culture et des activités en plein air à Marseille. 😊

```

---

## 📁 Structure du projet

```
cultural-events-rag/
├─ data/                        # Données CSV nettoyées des événements
├─ vector_store/                # Embeddings et index FAISS
│  ├─ events_embeddings.pkl
│  └─ faiss_index/
│      ├─ index.faiss
│      └─ metadata.pkl
├─ scripts/
│  ├─ step1_data_collection.py         # Collecte et création du dataset CSV
│  ├─ step2_rag_indexing.py   			# Chunking, génération d'embeddings et indexation FAISS
│  ├─ step3_rag_chatbot.py              # Chatbot interactif RAG utilisant Mistral + FAIS
├─ README.md                       # Documentation principale
├─ requirements.txt                 # Dépendances Python
├─ .env                             # Variables d'environnement (clé API)
├─ config.py                        # Configuration centralisée (paths, modèles, batch_size, etc.)


cultural-events-rag/
│
├─ data/                              # Données sources et nettoyées
│  └─ events_clean.csv
│
├─ vector_store/                      # Stockage vectoriel
│  ├─ events_embeddings.pkl           # Embeddings générés (Mistral)
│  └─ faiss_index/
│      ├─ index.faiss                 # Index FAISS (recherche vectorielle)
│      └─ metadata.pkl                # Métadonnées associées aux vecteurs
│
├─ scripts/                           # Pipelines principaux
│  ├─ __init__.py
│  ├─ step1_data_collection.py        # Etape 1: Collecte OpenAgenda + nettoyage + création du dataset CSV
│  ├─ step2_rag_indexing.py           # Etape 2: Chunking + embeddings + index FAISS
│  ├─ step3_rag_chatbot.py            # Etape 3: Chatbot RAG (retrieval + génération Mistral)
│  ├─ logger.py                       # Journalisation centralisé
│  └─ config.py                       # Configuration centralisée
│                                     # (paths, modèles, chunk_size, batch_size, etc.)
│
├─ tests/                             # Tests automatisés (pytest)
│  ├─ test_data_collection.py         # Tests unitaires étape 1
│  ├─ test_rag_indexing.py            # Tests unitaires étape 2
│  ├─ test_deduplication.py           # Test de déduplication sur les chunks
│  ├─ test_rag_chatbot.py             # Tests unitaires étape 3
│  ├─ test_logger.py                  # Journalisation avec rotation quotidienne 
│  └─ test_rag_evaluation.py          # Test d'évaluation de la pertinence du système RAG
│
├─ evaluation/                        # Évaluation globale du RAG
│  ├─ qa_dataset.json                 # Dataset de questions métier
│  ├─ test_rag_evaluation.py          # Interrogation du chatbot afin d'évaluer la qualité des réponses
│  └─ rag_eval_results.json           # Résultats de l'évaluation du RAG
│
├─ logs/                              # Logs d'exécution (optionnel mais recommandé)
│  └─ app.log                         # Logs pipeline + erreurs + debug
│
├─ pytest.ini                         # Configuration pour PyTest (tests unitaires et d’intégration)
│
├─ .env                               # Variables d’environnement (clé API Mistral)
│
├─ requirements.txt                   # Dépendances Python
│
├─ .gitignore                         # Liste des fichiers et dossiers à exclure du dépôt Git
│
└─ README.md                          # Documentation principale (GitHub)

```

---

## 🚀 Perspectives d'amélioration

- **Interface utilisateur web** - Développer une interface interactive via Streamlit pour faciliter l’accès au chatbot.
- **Enrichissement des données** - Ajouter de nouvelles sources d’événements culturels pour couvrir un périmètre géographique plus large et diversifier les types d’événements.
- **Support multi-langues** - Permettre la recherche et les réponses en plusieurs langues pour toucher un public plus large.
- **Amélioration des recommandations** - Affiner le pipeline RAG (chunking, embeddings, indexation) et le prompt pour augmenter la pertinence et la personnalisation des suggestions.

---

## 👤 Auteur

Sonia TACHAGO – Concevez et déployez un système RAG | OpenClassrooms

