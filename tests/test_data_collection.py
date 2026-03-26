import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.config import EXPECTED_CITY
# Import du script à tester
from scripts.step1_data_collection import (
    collect_events_from_open_agenda,
    create_dataframe,
    save_dataset
)

# -----------------------------
# MOCK DATA (simulation API)
# -----------------------------
TODAY = datetime.today()

MOCK_API_RESPONSE = {
    "results": [
        {
            "uid": "1",
            "title_fr": "Festival Jazz",
            "longdescription_fr": "Un grand festival de jazz",
            "description_fr": "Concerts en plein air",
            "location_city": "Marseille",
            "location_name": "Parc",
            "location_region": "PACA",
            "firstdate_begin": (TODAY - timedelta(days=30)).strftime("%Y-%m-%d"),  # récent
            "firstdate_end": (TODAY - timedelta(days=29)).strftime("%Y-%m-%d"),
            "location_tags": "musique"
        },
        {
            "uid": "2",
            "title_fr": "Ancien événement",
            "longdescription_fr": "Événement ancien",
            "description_fr": "Ancien",
            "location_city": "Marseille",
            "location_name": "Salle",
            "location_region": "IDF",
            "firstdate_begin": (TODAY - timedelta(days=500)).strftime("%Y-%m-%d"),  # ancien
            "firstdate_end": (TODAY - timedelta(days=499)).strftime("%Y-%m-%d"),
            "location_tags": "expo"
        }
    ]
}

# -----------------------------
# TEST 1 — Collect API (mock)
# -----------------------------
@patch("scripts.step1_data_collection.requests.get")
def test_collect_events(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_API_RESPONSE
    mock_response.raise_for_status.return_value = None

    mock_get.return_value = mock_response

    events = collect_events_from_open_agenda("fake_url", {})

    assert isinstance(events, list)
    assert len(events) == 2
    assert events[0]["uid"] == "1"


# -----------------------------
# TEST 2 — DataFrame création
# -----------------------------
def test_create_dataframe():
    events = MOCK_API_RESPONSE["results"]

    df = create_dataframe(events)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Vérifier renommage colonnes
    assert "event_id" in df.columns
    assert "title" in df.columns
    assert "long_description" in df.columns

    # Vérifier nettoyage
    assert "\n" not in df["long_description"].iloc[0]


# -----------------------------
# TEST 3 — Sauvegarde CSV
# -----------------------------
def test_save_dataset(tmp_path):
    df = pd.DataFrame({
        "event_id": ["1"],
        "title": ["Test Event"],
        "long_description": ["Description"]
    })

    output_file = tmp_path / "test_events.csv"

    save_dataset(df, output_file)

    assert output_file.exists()

    df_loaded = pd.read_csv(output_file)
    assert len(df_loaded) == 1
    assert df_loaded["title"].iloc[0] == "Test Event"



# -----------------------------
# TEST MÉTIER — événements récents (< 1 an)
# -----------------------------
def test_event_is_recent():
    events = MOCK_API_RESPONSE["results"]
    df = create_dataframe(events)

    # Conversion en datetime
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

    today = datetime.today()
    one_year_ago = today - timedelta(days=365)

    # Vérifier qu'au moins un événement est récent
    recent_events = df[df["start_date"] >= one_year_ago]

    assert len(recent_events) >= 1

    # Vérifier qu'on détecte bien un événement ancien
    old_events = df[df["start_date"] < one_year_ago]

    assert len(old_events) >= 1


# -----------------------------
# TEST MÉTIER — Ville attendue
# -----------------------------
def test_is_expected_city():
    """
    Vérifie que tous les événements appartiennent à la ville attendue.
    """

    events = [
        {
            "uid": "1",
            "title_fr": "EXPOSITION du concours d'idée",
            "longdescription_fr": "Une exposition interactive qui questionne la place de l’enfant dans la ville, avec une approche créative et accessible",
            "description_fr": "Marseille ville récréative",
            "location_city": "Marseille",
            "location_name": "Musée",
            "location_region": "PACA",
            "firstdate_begin": "2025-06-01",
            "firstdate_end": "2025-06-02",
            "location_tags": "art"
        },
        {
            "uid": "2",
            "title_fr": "Fun Radio Live",
            "longdescription_fr": "Un line-up exceptionnel avec des artistes",
            "description_fr": "Grand Concert Gratuit à Marseille",
            "location_city": "Marseille",
            "location_name": "¨Parc",
            "location_region": "PACA",
            "firstdate_begin": "2025-07-01",
            "firstdate_end": "2025-07-02",
            "location_tags": "musique"
        }
    ]

    df = create_dataframe(events)

    # expected_city = "Marseille"

    # Vérification
    assert not df.empty
    assert "city" in df.columns

    # Tous les événements doivent être dans la ville attendue
    assert all(df["city"].str.lower() == EXPECTED_CITY.lower())
    
def test_is_expected_city_with_wrong_data():
    """
    Vérifie que le test détecte des villes incorrectes.
    """

    events = [
        {
            "uid": "2",
            "title_fr": "Fun Radio Live",
            "longdescription_fr": "Un line-up exceptionnel avec des artistes",
            "description_fr": "Grand Concert Gratuit à Marseille",
            "location_city": "Marseille",
            "location_name": "¨Parc",
            "location_region": "PACA",
            "firstdate_begin": "2025-07-01",
            "firstdate_end": "2025-07-02",
            "location_tags": "musique"
        },
        {
            "uid": "2",
            "title_fr": "Expo Paris",
            "longdescription_fr": "Expo",
            "description_fr": "Musée",
            "location_city": "Paris",
            "location_name": "Musée",
            "location_region": "IDF",
            "firstdate_begin": "2025-07-01",
            "firstdate_end": "2025-07-02",
            "location_tags": "art"
        }
    ]

    df = create_dataframe(events)

    expected_city = "Marseille"

    # On vérifie qu'il y a au moins une anomalie
    assert any(df["city"].str.lower() != EXPECTED_CITY.lower())

