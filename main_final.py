from transformers import AutoModelForSequenceClassification, AutoTokenizer as AutoTokenizerSeq, \
    AutoModelForTokenClassification, AutoTokenizer, DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

from api.api import send_message, ask_RAG


# Charger les modèles et tokenizers pour NER et classification de texte
model_name_ner = "foucheta/nlp_esgi_td4_ner"
model_ner = AutoModelForTokenClassification.from_pretrained(model_name_ner)
tokenizer_ner = AutoTokenizer.from_pretrained(model_name_ner)

model_path = "model/distilbert_question_classifier"
model_classif = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer_classif = DistilBertTokenizerFast.from_pretrained(model_path)


# Étape 2: Fonction de prédiction NER
def predict_ner(sentence):
    # Tokeniser la phrase et obtenir les prédictions de label pour chaque token
    inputs = tokenizer_ner(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model_ner(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)

    # Convertir les ID de tokens en tokens et les prédictions en labels d'entité
    tokens = tokenizer_ner.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model_ner.config.id2label[prediction.item()] for prediction in predictions[0]]

    # Associer chaque token à son label et reconstruire les entités
    receiver = ""
    content = ""
    for token, label in zip(tokens, labels):
        if token in ["[CLS]", "[SEP]"]:
            continue
        if label == "LABEL_1":  # Receiver
            if token.startswith("##"):
                receiver += token[2:]
            else:
                receiver += " " + token
        elif label == "LABEL_2":  # Content
            if token.startswith("##"):
                content += token[2:]
            else:
                content += " " + token

    return receiver.strip(), content.strip()


# Étape 3: Fonction de classification de la requête
def classify_request(request):
    inputs = tokenizer_classif(request, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_classif(**inputs).logits
    predictions = torch.argmax(outputs, dim=1)

    return predictions.item()  # 0 pour 'question_rag', 1 pour 'send_message'


# Étape 4: Fonction principale pour traiter la requête
def send_virtual_assistant(request):
    # Classifier la requête
    task = classify_request(request)
    print("task = ", task)

    # Traiter en fonction du type de tâche
    if task == 1:  # send_message
        receiver, content = predict_ner(request)
        if receiver:
            return send_message(receiver, content)
        else:
            return "Receiver not found in the message."
    else:  # question_rag
        return ask_RAG(request)



# ======================= TEST # =======================
# Test cases
test_cases = [
    "Ask the python teacher when is the next class",
    "What are the pre-requisites for the python class?",
    "Send a message to John telling him about the meeting tomorrow.",
    "Tell Alice that the project deadline is next week.",
    "How do I install TensorFlow on my machine?",
    "Inform the students that the lecture starts at 10 AM."
]

# Execute tests
for test in test_cases:
    print(f"Request: {test}")
    print(send_virtual_assistant(test))
    print("-" * 50)  # Print a separator line
