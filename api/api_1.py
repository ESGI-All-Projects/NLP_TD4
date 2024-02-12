
from fastapi import FastAPI
from pydantic import BaseModel
import nest_asyncio
from threading import Thread

from NER

# Nécessaire pour exécuter un serveur async dans un notebook
nest_asyncio.apply()

app = FastAPI()

class Sentence(BaseModel):
    test_sentence: str

@app.post("/extract_entities/")
async def extract_entities_api(sentence: Sentence):
    # Placeholder pour la fonction `extract_entities` et ses dépendances
    # Vous devez définir `extract_entities`, `model`, `tokenizer`, et `label_map_inv` ici ou les importer si définis ailleurs
    extracted_entities = extract_entities(sentence.test_sentence, model, tokenizer, label_map_inv)
    result = {
        "job": "send_message",
        "receiver": " ".join(extracted_entities["person"]),
        "content": " ".join(extracted_entities["content"]),
    }
    return result

# Fonction pour exécuter le serveur Uvicorn dans un thread à part
def run_api():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Démarrer le serveur dans un thread à part pour ne pas bloquer le notebook
thread = Thread(target=run_api)
thread.start()
