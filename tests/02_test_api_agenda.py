import requests
import pandas as pd


url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

params = {
    "select": "title_fr, description_fr, longdescription_fr, firstdate_begin, location_city, location_region, location_tags, country_fr",
    # "select": "*",
    "where": "country_fr = 'Italie' AND firstdate_begin >= '2025-01-01'", 
    # "where": "location_countrycode = 'FR' AND firstdate_begin >= '2025-01-01'", 
    # "limit": 50
}

response = requests.get(url, params=params)
# response = requests.get(url)

# afficher le statut
print("Status code:", response.status_code)

# convertir la réponse en JSON
data = response.json()

df = pd.DataFrame(data["results"])

print(df.head())
