import requests
from pathlib import Path
import pandas as pd
import json


API_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"


PARAMS = {
    "select": "uid, title_fr, description_fr, longdescription_fr, location_city, location_region, firstdate_begin, firstdate_end, location_tags",
    "where": "description_fr IS NOT NULL AND location_city = 'Marseille' AND firstdate_begin >= '2025-01-01'",
    "order_by": "firstdate_begin",
    "limit": 100
}

# params = {
            # "select": "*",
            # "where": 'firstdate_begin >= "2024-07-01"',
            # "limit": limit,
            # "offset": offset,
            # "order_by": "location_city, firstdate_begin",
            # "refine": 'location_region:"Occitanie"',
        # }

OUTPUT_PATH = Path("data/events_clean.csv")
MD_OUTPUT_DIR = Path("data/docs_md")


def collect_events_from_open_agenda(api_url, params):
    """
    Récupère les événements via l'API OpenAgenda.
    """
    print("Collecte des événements depuis OpenAgenda...")

    response = requests.get(api_url, params=params)
    response.raise_for_status()

    data = response.json()
    events = data.get("results", [])

    print(f"{len(events)} événements collectés")

    return events


def create_dataframe(events):
    """
    Convertit les résultats JSON en DataFrame.
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

    # Nettoyage
    df = df.dropna(subset=["description"])
    df["description"] = df["description"].str.replace("\n", " ", regex=False)

    df = df.dropna(subset=["long_description"])
    df["long_description"] = df["long_description"].str.replace("\n", " ", regex=False)

    return df


def save_dataset(df):
    """
    Sauvegarde le dataset nettoyé en CSV.
    """

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset sauvegardé : {OUTPUT_PATH}")


def clean_save_events_md(df, output_dir):
    """
    Sauvegarde chaque événement en fichier Markdown lisible.
    """

    print("🔄 Génération des fichiers Markdown...")

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():

        event_id = row.get("event_id", "Sans titre")
        title = row.get("title", "Sans titre")
        city = row.get("city", "Inconnu")
        venue = row.get("venue", "Lieu inconnu")
        region = row.get("region", "Region inconnu")
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

    print(f" {len(df)} fichiers Markdown générés dans {output_dir}")


def main():

    print(" Récupération des événements...")

    events = collect_events_from_open_agenda(API_URL, PARAMS)

    df = create_dataframe(events)

    print(f"{len(df)} événements après nettoyage")

    save_dataset(df)

    # Génération des documents Markdown (optionnel)
    clean_save_events_md(df, MD_OUTPUT_DIR)


if __name__ == "__main__":
    main()
