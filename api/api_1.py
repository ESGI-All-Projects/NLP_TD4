
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
    # import `extract_entities`, `model`, `tokenizer`, et `label_map_inv`
    extracted_entities = extract_entities(sentence.test_sentence, model, tokenizer, label_map_inv)
    result = {
        "job": "send_message",
        "receiver": " ".join(extracted_entities["person"]),
        "content": " ".join(extracted_entities["content"]),
    }
    return result

def run_api():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

thread = Thread(target=run_api)
thread.start()
