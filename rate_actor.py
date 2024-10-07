"""
File: concat_names.py
Author: Nischith Bairannanavara Omprakash
Date: April 23, 2024
Description: Main file to generate ranking of actors in news article.
"""
import os
import time
import gensim.downloader as api
import gensim
import argparse
from annotation_generator import *
from noun_actor_list import *
from concat_names import *
from dependency_tree_parser import *
from actor_action_mapping import *
from bert_model_action_actor_predictor import *
from impact_score_ranking import *
import logging

model_name = 'word2vec-google-news-300'
NER_List = ['ORG', "GPE", "PRODUCT", "NORP", "LOC", "FAC", "LAW", 'PERSON', 'EVENT']
# model = gensim.downloader.load(model_name)

logging.basicConfig(filename='gensim_model_loading.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s')

model_name = 'word2vec-google-news-300'  # Example model name
model = None
try:
    model = api.load(model_name)
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error("Failed to load the model due to an error: %s", e)
    print(f"An error occurred: {e}")
    model = None


nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')


def rate_actor(text):
    doc_original = nlp(text)
    actor_noun = []
    for tok in doc_original:
        if tok.pos_ == "NOUN" and (tok.ent_type_ in NER_List or not tok.ent_type_) and is_noun_word_an_actor(tok.text):
            actor_noun.append(tok.text)
    new_noun_actor_list = filter_actor_nouns(actor_noun, model)
    new_noun_actor_list = filter_invalid_noun(new_noun_actor_list)
    concat_text, new_concatenated_names = concat_pronoun(doc_original)
    doc_concat = nlp(concat_text)
    verb_mapping = map_verb_actor(doc_concat, new_noun_actor_list)
    all_actors = get_all_actors(doc_concat, new_noun_actor_list)
    actor_action_spacy_mapping = map_actor_to_action(all_actors, verb_mapping, doc_concat)
    actor_action_spacy_mapping = merge_repeated_names(actor_action_spacy_mapping, new_concatenated_names)
    bert_mapping = get_bert_model_prediction(doc_original)
    bert_mapping = merge_repeated_names(bert_mapping, new_concatenated_names)
    final_actor_action = merge_bert_spacy_mapping(actor_action_spacy_mapping, bert_mapping)
    get_ranking(final_actor_action)
    get_graph(final_actor_action)
    time.sleep(360)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Script to rate actors in news article")

    # Add an argument
    parser.add_argument('--path', type=str, help='News article file path')

    # Parse the arguments
    args = parser.parse_args()
    file_content = ""
    article_path = args.path
    try:
        with open(article_path, 'r', encoding='utf-8') as file:
            # Read the entire content of the file
            file_content = file.read()
    except Exception as err:
        print(f"Error reading article file: {err}")
    rate_actor(file_content)


if __name__ == "__main__":
    main()
