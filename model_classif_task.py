# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
# from datasets import Dataset
#
# # Charger les données
# data_path = "data/question_classif.csv"
# df = pd.read_csv(data_path)
# df['label'] = df['label'].astype(int)
#
# # Préparer les datasets
# train_df, test_df = train_test_split(df, test_size=0.2)
# train_dataset = Dataset.from_pandas(train_df)
# test_dataset = Dataset.from_pandas(test_df)
#
# # Tokenization
# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#
# def tokenize_function(examples):
#     return tokenizer(examples["question"], padding="max_length", truncation=True)
#
# tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
# tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
#
# # Modèle
# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
#
# # Entraînement
# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     evaluation_strategy="epoch",  # Modifié ici
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_test_dataset,
# )
#
# trainer.train()
#
# # Sauvegarder le modèle
# model_path = "distilbert_question_classifier"
# model.save_pretrained(model_path)
# tokenizer.save_pretrained(model_path)
#

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset, load_metric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer
import numpy as np

# Charger les données
df = pd.read_csv('data/question_classif.csv')
df['label'] = df['label'].astype(int)

print(df['label'].value_counts())

# Préparer les datasets
train_df, test_df = train_test_split(df, test_size=0.2)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['question'], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Modèle
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Geler toutes les couches sauf les couches feed-forward de la dernière couche cachée
for name, param in model.named_parameters():
    if 'classifier' not in name: # classifier layer
        param.requires_grad = False


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Imprimer les premiers quelques logits et labels pour débogage
    print("Quelques logits prédits:", logits[:5])
    print("Quelques labels réels:", labels[:5])
    print("Quelques prédictions:", predictions[:5])

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,  # Augmenter la taille du batch peut aider à la généralisation
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.02,
    evaluation_strategy="epoch",  # Évaluer à la fin de chaque époque
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # Ajout de la fonction de calcul des métriques
)


trainer.train()

evaluation_results = trainer.evaluate()
print(evaluation_results)

# Sauvegarder le modèle
model.save_pretrained('model/distilbert_question_classifier')
tokenizer.save_pretrained('model/distilbert_question_classifier')
