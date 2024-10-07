"""
File: run_annotation_generator.py
Author: Nischith Bairannanavara Omprakash
Date: April 23, 2024
Description: Contains method to run all methods needed to generate annotations.
"""
import os
import csv
import gensim.downloader as api
import json
import spacy
from annotation_generator import *
from noun_actor_list import *
from concat_names import *
from dependency_tree_parser import *


def is_model_downloaded(model_name):
    # Check if the model is in the list of downloaded models
    return model_name in api.info()['models']

model_name = 'word2vec-google-news-300'

if not is_model_downloaded(model_name):
    model = api.load(model_name)  # This will download and load the model
else:
    model = api.load(model_name, return_path=False)  # Load the model without downloading

nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')
NER_List = ['ORG', "GPE", "PRODUCT", "NORP", "LOC", "FAC", "LAW", 'PERSON', 'EVENT']
file_path = "bloomberg_quint_news.json"
# Open the file in read mode
config = None
config_file = "config.json"
# Open the file in read mode


def main():

    with open(config_file, "r") as file:
        # Load the JSON data
        config = json.load(file)["annotation"]

    with open(file_path, "r") as file:
        # Load the JSON data
        data = json.load(file)
    # Specify the name of the file to check and create
    output_file = config["output_annotation_file_name"]
    # Check if the file exists
    if not os.path.isfile(output_file):
        # File does not exist, create it
        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Sentence', 'Tokens', 'actor_labels', 'action_labels'])
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        article_range = eval(config["article_range_to_annotate"])
        for article in data[article_range[0]:article_range[1]]:
            original_text = article.get("description")
            doc_original = nlp(original_text)
            actor_noun = []
            for tok in doc_original:
                if tok.pos_ == "NOUN" and (tok.ent_type_ in NER_List or not tok.ent_type_) and is_noun_word_an_actor(tok.text):
                    actor_noun.append(tok.text)
            new_actor_list = filter_actor_nouns(actor_noun, model)
            new_actor_list = filter_invalid_noun(new_actor_list)
            concat_text, new_concatenated_names = concat_pronoun(doc_original)
            doc_concat = nlp(concat_text)
            verb_mapping = map_verb_actor(doc_concat, new_actor_list)
            annotations = perform_annotations(doc_concat, verb_mapping, new_actor_list)
            for annotation in annotations:
                writer.writerow(annotation)


if __name__ == "__main__":
    main()



