"""
01_collect_events.py

Pipeline de collecte, nettoyage et export d'événements culturels depuis OpenAgenda.
Les données sont sauvegardées au format CSV et, en option, en fichiers Markdown lisibles.
"""


from pathlib import Path
import requests
import pandas as pd
from config import API_URL, PARAMS, OUTPUT_PATH, MD_OUTPUT_DIR


def collect_events_from_open_agenda(api_url: str, params: dict) -> list[dict]:
    """
    Récupère les événements depuis l'API OpenAgenda.
    
    Args:
        api_url (str): URL de l'API OpenAgenda.
        params (dict): Paramètres de requête (select, where, order_by, limit).
    
    Returns:
        list[dict]: Liste des événements sous forme de dictionnaires JSON.
    """
    print("Collecte des événements depuis OpenAgenda...")
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    data = response.json()
    events = data.get("results", [])
    print(f"\t{len(events)} événements collectés")
    return events


def create_dataframe(events: list[dict]) -> pd.DataFrame:
    """
    Transforme la liste d'événements JSON en DataFrame pandas et nettoie les colonnes.
    
    Args:
        events (list[dict]): Liste d'événements collectés.
    
    Returns:
        pd.DataFrame: DataFrame nettoyée avec colonnes standardisées.
    """
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

    return df


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """
    Sauvegarde le dataset nettoyé au format CSV.
    
    Args:
        df (pd.DataFrame): DataFrame à sauvegarder.
        output_path (Path): Chemin du fichier CSV de sortie.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\tDataset sauvegardé : {output_path}")


def clean_save_events_md(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Génère un fichier Markdown par événement.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les événements.
        output_dir (Path): Répertoire de sortie pour les fichiers Markdown.
    """
    print("\tGénération des fichiers Markdown...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        event_id = row.get("event_id", "Sans titre")
        title = row.get("title", "Sans titre")
        city = row.get("city", "Inconnu")
        venue = row.get("venue", "Lieu inconnu")
        region = row.get("region", "Région inconnue")
        date = row.get("start_date", "Date inconnue")
        description = row.get("description", "")
        long_description = row.get("long_description", "")
        tags = row.get("tags", "")

        md_content = f"""# {title}

            - Event_id : {event_id}
            - Type évènement : {tags}
            - Titre : {title}
            - Date : {date}
            - Ville : {city}
            - Lieu : {venue}
            - Région : {region}

            ## Description Complète
            {long_description}
        """

        filename = output_dir / f"event_{idx+1}.md"
        with open(filename, "w", encoding="utf-8") as f_md:
            f_md.write(md_content)

    print(f"\t{len(df)} fichiers Markdown générés dans {output_dir}")


def main() -> None:
    """
    Pipeline principal :
    1. Collecte des événements
    2. Nettoyage et transformation en DataFrame
    3. Sauvegarde CSV
    4. Génération optionnelle des fichiers Markdown
    """
    print("Démarrage du pipeline de collecte d'événements...")

    events = collect_events_from_open_agenda(API_URL, PARAMS)
    df = create_dataframe(events)

    print(f"\t{len(df)} événements après nettoyage")

    save_dataset(df, OUTPUT_PATH)
    clean_save_events_md(df, MD_OUTPUT_DIR)


if __name__ == "__main__":
    main()
