#
#
# label_map = {"person": 1, "content": 2}
# def map_labels(label_list):
#     return [label_map.get(label, 0) for label in label_list]
#
#
#
# # ===========================
# # Initialiser le tokeniseur
# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#
# # Fonction de tokenisation et d'alignement des labels
# def tokenize_and_align_labels(examples):
#     tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True)
#     labels = []
#     for i, label in enumerate(examples["labels"]):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)
#         label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
#         labels.append(label_ids)
#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs
#
#
#
#
# # taille maximale des séquences
# max_length = 128
#
# # fonction pour aligner et tronquer les séquences
# def align_and_truncate(data):
#     input_ids = tf.keras.preprocessing.sequence.pad_sequences(
#         data['input_ids'], maxlen=max_length, dtype='long', padding='post', truncating='post'
#     )
#     attention_mask = tf.keras.preprocessing.sequence.pad_sequences(
#         data['attention_mask'], maxlen=max_length, dtype='long', padding='post', truncating='post'
#     )
#     labels = tf.keras.preprocessing.sequence.pad_sequences(
#         data['labels'], maxlen=max_length, dtype='long', padding='post', truncating='post', value=-100
#     )
#     return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels
#
#
#
#
# def custom_loss(y_true, y_pred):
#     # Trouver les indices où les labels ne sont pas -100
#     active_loss = tf.reshape(y_true, (-1,)) != -100
#
#     # Sélectionner les prédictions et les labels réels pour ces indices
#     reduced_logits = tf.boolean_mask(tf.reshape(y_pred, (-1, tf.shape(y_pred)[2])), active_loss)
#     reduced_labels = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)
#
#     loss = tf.keras.losses.sparse_categorical_crossentropy(reduced_labels, reduced_logits, from_logits=True)
#     return loss
#
#
#
# def align_predictions(predictions, label_ids):
#     preds = np.argmax(predictions, axis=2)
#
#     batch_size, seq_len = preds.shape
#
#     out_label_list = [[] for _ in range(batch_size)]
#     preds_list = [[] for _ in range(batch_size)]
#
#     for i in range(batch_size):
#         for j in range(seq_len):
#             if label_ids[i, j] != -100:
#                 out_label_list[i].append(label_map_inv[label_ids[i][j]])
#                 preds_list[i].append(label_map_inv[preds[i][j]])
#
#     return preds_list, out_label_list
#
#
#
#
# def extract_entities(sentence, model, tokenizer, label_map_inv):
#     # Tokeniser la phrase
#     inputs = tokenizer.encode_plus(sentence, return_tensors='tf', max_length=128, pad_to_max_length=True)
#
#     # Obtenir les prédictions du modèle
#     predictions = model.predict({'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']})[0]
#
#     # Obtenir les indices des labels prédits
#     predicted_label_indices = tf.argmax(predictions, axis=-1).numpy()[0]
#
#     # Extraire les labels et les tokens correspondants
#     tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].numpy()[0])
#     labels = [label_map_inv.get(idx, "O") for idx in predicted_label_indices]
#
#     # Ignorer les tokens spéciaux et les paddings
#     tokens_labels = [(token, label) for token, label in zip(tokens, labels) if token not in ["[CLS]", "[SEP]", "[PAD]"]]
#
#     # Extraire les entités
#     entities = {"person": [], "content": []}
#     for token, label in tokens_labels:
#         if label == "person":
#             entities["person"].append(token)
#         elif label == "content":
#             entities["content"].append(token)
#
#     return entities


