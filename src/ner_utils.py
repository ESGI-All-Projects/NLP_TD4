import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, TFDistilBertForTokenClassification
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import nest_asyncio
from threading import Thread
import uvicorn


# Load data from a CSV file
def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    df['words'] = df['words'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df['labels'] = df['labels'].apply(lambda x: map_labels(eval(x)) if isinstance(x, str) else x)
    return df


# Map textual labels to integers
label_map = {"person": 1, "content": 2}


def map_labels(label_list):
    return [label_map.get(label, 0) for label in label_list]


# Create a Hugging Face Dataset
def create_dataset(df):
    return Dataset.from_pandas(df)


# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# Tokenization and label alignment function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Extract entities from text
def extract_entities(text, model, tokenizer, label_map_inv):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(inputs)
    predictions = tf.argmax(outputs.logits, axis=-1)
    predicted_label_indices = predictions.numpy()[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].numpy()[0])
    labels = [label_map_inv.get(idx, "O") for idx in predicted_label_indices]

    tokens_labels = [(token, label) for token, label in zip(tokens, labels) if token not in ["[CLS]", "[SEP]", "[PAD]"]]

    entities = {"person": [], "content": []}
    for token, label in tokens_labels:
        if label == "person":
            entities["person"].append(token)
        elif label == "content":
            entities["content"].append(token)

    return entities


# # FastAPI app for entity extraction
# nest_asyncio.apply()
# app = FastAPI()
#
#
# class Sentence(BaseModel):
#     test_sentence: str
#
#
# @app.post("/extract_entities/")
# async def extract_entities_api(sentence: Sentence):
#     extracted_entities = extract_entities(sentence.test_sentence, model, tokenizer, label_map_inv)
#     result = {
#         "job": "send_message",
#         "receiver": " ".join(extracted_entities["person"]),
#         "content": " ".join(extracted_entities["content"]),
#     }
#     return result


# # Function to run the API server in a separate thread
# def run_api():
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


import os
from transformers import TFTrainingArguments, TFTrainer


def train_model(tokenized_data, model, epochs=3, batch_size=8):
    # Define the output directory for saving model results
    output_dir = './results'  # Adjust this path as needed

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    training_args = TFTrainingArguments(
        output_dir=output_dir,  # Output directory
        num_train_epochs=epochs,  # Total number of training epochs
        per_device_train_batch_size=batch_size,  # Batch size per device during training
        per_device_eval_batch_size=batch_size,  # Batch size for evaluation
        warmup_steps=500,  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # Strength of weight decay
    )

    trainer = TFTrainer(
        model=model,  # The instantiated 🤗 Transformers model to be trained
        args=training_args,  # Training arguments, defined above
        train_dataset=tokenized_data,  # Training dataset
    )

    trainer.train()



# # Function to train the model
# def train_model(tokenized_data, model, epochs=3, batch_size=8):
#     import os
#
#     training_args = TFTrainingArguments(
#         output_dir='./results',  # output directory
#         num_train_epochs=epochs,  # total number of training epochs
#         per_device_train_batch_size=batch_size,  # batch size per device during training
#         per_device_eval_batch_size=batch_size,  # batch size for evaluation
#         warmup_steps=500,  # number of warmup steps for learning rate scheduler
#         weight_decay=0.01,  # strength of weight decay
#         logging_dir='./logs',  # directory for storing logs
#         logging_steps=10,
#     )
#
#     trainer = TFTrainer(
#         model=model,  # the instantiated 🤗 Transformers model to be trained
#         args=training_args,  # training arguments, defined above
#         train_dataset=tokenized_data,  # training dataset
#     )
#
#     # Ensure the logging directory exists
#     logging_dir = training_args.logging_dir
#     if not os.path.exists(logging_dir):
#         os.makedirs(logging_dir, exist_ok=True)
#
#     trainer.train()


# Function to save the trained model
def save_model(model, save_path):
    model.save_pretrained(save_path)
