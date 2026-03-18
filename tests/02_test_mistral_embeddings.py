

from mistralai import Mistral

client = Mistral(api_key="3IDz1gqFwh7e9IWYcyp0hFiCpiMEIxwi")
response = client.embeddings.create(
    model="mistral-embed",
    inputs=["Bonjour", "Festival de jazz"]  # ✅ utiliser 'inputs' ici
)

print(response.data[0].embedding[:5])