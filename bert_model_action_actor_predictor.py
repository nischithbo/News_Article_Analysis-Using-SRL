"""
File: bert_model_action_actor_prediction.py
Author: Nischith Bairannanavara Omprakash
Date: April 23, 2024
Description: Main file to generate ranking of actors in news article using BERT model.
"""
import torch
import copy
import json
from transformers import RobertaTokenizer, RobertaForTokenClassification
import spacy
import numpy as np
from concat_names import *

config_file = "config.json"
with open(config_file, "r") as file:
        # Load the JSON data
        config = json.load(file)["testing_config"]

nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')
N_ACTORS = int(config["N_ACTOR"])
save_model = config["save_model_path"]
save_tokenizer = config["save_tokenizer_path"]
max_length_sentence = config["MAX_SENTENCE_LENGTH"]

device = torch.device("cpu")
bert_tokenizer = RobertaTokenizer.from_pretrained(save_tokenizer)
model = RobertaForTokenClassification.from_pretrained(save_model)

NER_List = ['ORG', "GPE", "PRODUCT", "NORP", "LOC", "FAC", "LAW", 'PERSON', 'EVENT']


def replace_pronoun(text):
    doc = nlp(text)
    new_text = ""
    for i in range(0, len(doc)):
        if doc._.coref_chains.resolve(doc[i]):
            nouns = doc._.coref_chains.resolve(doc[i])
            if len(nouns) == 1:
                new_text += nouns[0].text + " "
                # print(doc[i].text, nouns[0].text_with_ws)
                continue
            else:
                name = ""
                for j in range(0, len(nouns)):
                    if not name:
                        name += nouns[j].text + " "
                    elif j == len(nouns) - 1:
                        name += " and " + nouns[j].text + " "
                    else:
                        name += ", " + nouns[j].text
            # print(doc[i].text, name)
            new_text += name
        else:
            new_text += doc[i].text_with_ws
    return new_text


def get_all_actor_list(doc_):
    all_actors = []
    for tok in doc_:
        if tok.pos_ == "PROPN" and (tok.ent_type_ in NER_List or not tok.ent_type_):
            all_actors.append(tok.text)
        elif tok.pos_ == "NOUN" and (tok.ent_type_ in NER_List or not tok.ent_type_):
            all_actors.append(tok.text)
    return all_actors


def get_all_verb(doc_):
    all_verbs = []
    for tok in doc_:
        if tok.pos_ == "VERB":
            all_verbs.append(tok.text)
    return all_verbs


def generate_final_model_actor_action_mapping(actor_action_mapping):
    final_mapping = {}
    for each_sentence_mapping in actor_action_mapping:
        if each_sentence_mapping:
            for each_actor in each_sentence_mapping:
                # print(each_sentence_mapping[each_actor])
                if each_sentence_mapping[each_actor][0] in final_mapping:
                    final_mapping[each_sentence_mapping[each_actor][0]][0] += each_sentence_mapping[each_actor][1]
                    final_mapping[each_sentence_mapping[each_actor][0]][1].extend(each_sentence_mapping[each_actor][2])
                else:
                    final_mapping[each_sentence_mapping[each_actor][0]] = [each_sentence_mapping[each_actor][1], each_sentence_mapping[each_actor][2]]
    return final_mapping


def get_bert_model_prediction(doc):
    concat_text, concatenated_names = concat_pronoun(doc)
    resolved_text = replace_pronoun(concat_text)
    doc = nlp(resolved_text)
    all_actors = get_all_actor_list(doc)
    all_verbs = get_all_verb(doc)
    sentences_token = []
    for sentence in doc.sents:
        sent_token = []
        for tok in sentence:
            sent_token.append(tok.text)
        sentences_token.append(sent_token)
    # print(sentences_token)
    article_pre_final_mapping = []
    for each_sentence in sentences_token:
        # attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        token_ids = [bert_tokenizer.convert_tokens_to_ids(
            token) if token in bert_tokenizer.get_vocab() else bert_tokenizer.unk_token_id for token in each_sentence]
        token_ids = [bert_tokenizer.cls_token_id] + token_ids + [bert_tokenizer.sep_token_id]
        pad_length = max_length_sentence + 2 - len(token_ids)
        attention_mask = [1] * len(token_ids)
        token_ids += [bert_tokenizer.pad_token_id] * pad_length
        attention_mask += [0] * pad_length  # Attention mask
        input_ids_tensor = torch.tensor([token_ids]).to(device)
        attention_mask_tensor = torch.tensor([attention_mask]).to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
            logits = outputs.logits
        predictions = torch.sigmoid(logits)
        predictions = predictions.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
        # print(predictions[1])
        threshold = 0.4
        actor_mapping = {}
        binary_predictions = (predictions > threshold).astype(int)
        for i, (token, actor_prediction, verb_prediction) in enumerate(
                zip(bert_tokenizer.convert_ids_to_tokens(token_ids), predictions[:, :N_ACTORS],
                    predictions[:, N_ACTORS:])):
            if token not in ['<s>', '<pad>', '</s>']:  # Skipping specific tokens
                actor_threshold = 0.35
                binary_prediction = (actor_prediction > actor_threshold).astype(int)
                if 1 in binary_prediction:
                    prediction_list = actor_prediction.tolist()
                    max_element = max(prediction_list)
                    max_index = prediction_list.index(max_element)
                    if max_index in actor_mapping and each_sentence[i - 1] in all_actors:
                        if each_sentence[i - 1] == actor_mapping[max_index][0]:
                            actor_mapping[max_index][1] += 1
                        else:
                            actor_mapping[max_index][0] = each_sentence[i - 1]
                            actor_mapping[max_index][1] = 1
                            actor_mapping[max_index][2] = []
                    elif max_index not in actor_mapping and each_sentence[i - 1] in all_actors:
                        actor_mapping[max_index] = [each_sentence[i - 1], 1, []]
            if token == '</s>':
                break
        for i, (token, actor_prediction, verb_prediction) in enumerate(
                zip(bert_tokenizer.convert_ids_to_tokens(token_ids), predictions[:, :N_ACTORS],
                    predictions[:, N_ACTORS:])):
            if token not in ['<s>', '<pad>', '</s>']:  # Skipping specific tokens
                verb_threshold = 0.25
                binary_prediction = (verb_prediction > verb_threshold).astype(int)
                if 1 in binary_prediction and each_sentence[i - 1] in all_verbs:
                    action_prediction_list = binary_prediction.tolist()
                    for index, prediction in enumerate(action_prediction_list):
                        if prediction == 1:
                            if index in actor_mapping:
                                actor_mapping[index][2].append(each_sentence[i - 1])

        article_pre_final_mapping.append(actor_mapping)

    return generate_final_model_actor_action_mapping(article_pre_final_mapping)
