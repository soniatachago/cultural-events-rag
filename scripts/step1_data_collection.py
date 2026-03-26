"""
step1_data_collection.py

Pipeline de collecte, nettoyage et export d'événements culturels depuis OpenAgenda.
Les données sont sauvegardées au format CSV et, en option, en fichiers Markdown lisibles.
"""


from pathlib import Path
import requests
import pandas as pd
import logging
from scripts.config import LOGGER, MAX_RETRIES, TIMEOUT, API_URL, PARAMS, OUTPUT_PATH, MD_OUTPUT_DIR


# -----------------------------
# COLLECTE API
# -----------------------------
def collect_events_from_open_agenda(api_url: str, params: dict) -> list[dict]:
    """
    Récupère les événements depuis l'API OpenAgenda.
    
    Args:
        api_url (str): URL de l'API OpenAgenda.
        params (dict): Paramètres de requête (select, where, order_by, limit).
    
    Returns:
        list[dict]: Liste des événements sous forme de dictionnaires JSON.
    """
    LOGGER.info("\n Collecte des événements depuis OpenAgenda...")
    
    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = requests.get(api_url, params=params, timeout=TIMEOUT)
            response.raise_for_status()

            data = response.json()
            events = data.get("results", [])

            LOGGER.info(f"✅ {len(events)} événements collectés")
            return events

        except requests.exceptions.Timeout:
            LOGGER.warning(f"⏱️ Timeout API (tentative {attempt}/{MAX_RETRIES})")

        except requests.exceptions.HTTPError as e:
            LOGGER.error(f"❌ Erreur HTTP: {e}")
            break  # inutile de retry si 4xx/5xx critique

        except Exception as e:
            LOGGER.error(f"❌ Erreur inattendue: {e}")

        # retry avec backoff simple
        time.sleep(2 * attempt)

    LOGGER.error("❌ Échec de la collecte après plusieurs tentatives")
    return []


# -----------------------------
# DATAFRAME
# -----------------------------
def create_dataframe(events: list[dict]) -> pd.DataFrame:
    """
    Transforme la liste d'événements JSON en DataFrame pandas et nettoie les colonnes.
    
    Args:
        events (list[dict]): Liste d'événements collectés.
    
    Returns:
        pd.DataFrame: DataFrame nettoyée avec colonnes standardisées.
    """

    if not events:
        LOGGER.warning("⚠️ Aucun événement à transformer")
        return pd.DataFrame()

    try:
        df = pd.DataFrame(events)

        df = df.rename(columns={
            "uid": "event_id",
            "title_fr": "title",
            "longdescription_fr": "long_description",
            "description_fr": "description",
            "location_city": "city",
            "location_name": "venue",
            "location_region": "region",
            "firstdate_begin": "start_date",
            "firstdate_end": "end_date",
            "location_tags": "tags"
        })

        # Nettoyage des champs texte
        df = df.dropna(subset=["description"])
        df["description"] = df["description"].str.replace("\n", " ", regex=False)

        df = df.dropna(subset=["long_description"])
        df["long_description"] = df["long_description"].str.replace("\n", " ", regex=False)

        LOGGER.info(f"✅ DataFrame créée ({len(df)} événements)")

        return df
    
    except Exception as e:
        LOGGER.error(f"❌ Erreur création DataFrame: {e}")
        return pd.DataFrame()


# -----------------------------
# SAUVEGARDE DATASET CSV
# -----------------------------
def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """
    Sauvegarde le dataset nettoyé au format CSV.
    
    Args:
        df (pd.DataFrame): DataFrame à sauvegarder.
        output_path (Path): Chemin du fichier CSV de sortie.
    """
    
    if df.empty:
        LOGGER.warning("⚠️ DataFrame vide - aucun fichier sauvegardé")
        return

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        LOGGER.info(f"💾 Dataset sauvegardé dans {output_path}")

    except Exception as e:
        LOGGER.error(f"❌ Erreur sauvegarde CSV: {e}")


# -----------------------------
# SAUVEGARDE SOUS MARKDOWN
# -----------------------------
def clean_save_events_md(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Génère un fichier Markdown par événement.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les événements.
        output_dir (Path): Répertoire de sortie pour les fichiers Markdown.
    """

    if df.empty:
        LOGGER.warning("⚠️ Aucun événement pour génération Markdown")
        return

    LOGGER.info("📝 Génération des fichiers Markdown...")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in df.iterrows():

            md_content = f"""# {row.get("title", "Sans titre")}

                - Event_id : {row.get("event_id", "")}
                - Type évènement : {row.get("tags", "")}
                - Date de début : {row.get("start_date", "")}
                - Date de fin : {row.get("end_date", "")}
                - Ville : {row.get("city", "")}
                - Lieu : {row.get("venue", "")}
                - Région : {row.get("region", "")}

                ## Description Complète
                {row.get("long_description", "")}
            """

            filename = output_dir / f"event_{idx+1}.md"

            with open(filename, "w", encoding="utf-8") as f:
                f.write(md_content)

        LOGGER.info(f"✅ {len(df)} fichiers Markdown générés dans {output_dir}")

    except Exception as e:
        LOGGER.error(f"❌ Erreur génération Markdown: {e}")

    
# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    """
    Pipeline principal :
    1. Collecte des événements
    2. Nettoyage et transformation en DataFrame
    3. Sauvegarde CSV
    4. Génération optionnelle des fichiers Markdown
    """
    
    LOGGER.info("🚀 Démarrage pipeline collecte d'événements")
    
    try:
        events = collect_events_from_open_agenda(API_URL, PARAMS)

        if not events:
            LOGGER.error("❌ Pipeline interrompu (aucun événement)")
            return

        df = create_dataframe(events)

        if df.empty:
            LOGGER.error("❌ Pipeline interrompu (DataFrame vide)")
            return

        save_dataset(df, OUTPUT_PATH)
        clean_save_events_md(df, MD_OUTPUT_DIR)

        LOGGER.info("\n🎯 Pipeline terminé avec succès")

    except Exception as e:
        LOGGER.critical(f"💥 Erreur critique pipeline: {e}")



if __name__ == "__main__":
    main()
