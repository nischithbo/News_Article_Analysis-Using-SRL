"""
File: noun_actor_list.py
Author: Nischith Bairannanavara Omprakash
Date: April 23, 2024,
Description: Contains methods to get noun in the text which act as actors.
"""
import nltk


# Ensure that necessary data is downloaded
if not nltk.download('wordnet', quiet=True):  # The quiet=True option suppresses output if the download is successful
    # If 'wordnet' is not downloaded, it will download here
    nltk.download('wordnet')

from nltk.corpus import wordnet as wn
NER_List = ['ORG', "GPE", "PRODUCT", "NORP", "LOC", "FAC", "LAW", 'PERSON', 'EVENT']


def filter_invalid_noun(noun_list):
    invalid = [
    "Mr.", "Mrs.", "Ms.", "Mx.", "Dr.", "Prof.", "Rev.", "Fr.", "Sr.", "Br.",
    "Capt.", "Sgt.", "Lt.", "Gen.", "Hon.", "Pres.", "Gov.", "Atty.", "Dean", "Chancellor", "Rector",
    ".", ",", "!", "?", ":", ";", "-", "--", "'", "\"", "(", ")", "[", "]", "{", "}",
    "/", "\\", "@", "#", "$", "%", "^", "&", "*", "_", "~", "`", "|", "…", "men", "woman", "man"
    ]
    return [noun for noun in noun_list if noun not in invalid]


def is_noun_word_an_actor(noun):
    # Initialize the 'actor' category with basic human-related synsets
    actor_synsets = set(wn.synsets('person', pos=wn.NOUN))
    actor_synsets.update(wn.synsets('human', pos=wn.NOUN))
    actor_synsets.update(wn.synsets('men', pos=wn.NOUN))
    actor_synsets.update(wn.synsets('officials', pos=wn.NOUN))
    actor_synsets.update(wn.synsets('organization', pos=wn.NOUN))
    actor_synsets.update(wn.synsets('criminals', pos=wn.NOUN))
    actor_synsets.update(wn.synsets('woman', pos=wn.NOUN))

    synsets = wn.synsets(noun, pos=wn.NOUN)

    # Check for matching hypernyms in the noun's synsets
    for synset in synsets:
        # Get all hypernym paths for this synset
        hypernym_paths = synset.hypernym_paths()

        # Check each path to see if it intersects with the actor synsets
        for path in hypernym_paths:
            if any(hypernym in path for hypernym in actor_synsets):
                return True  # Return 'actor' if any of the paths match the actor_synsets

    return False  # Return 'unknown' if no matches are found


def filter_actor_nouns(actor_list, model):
    filtered_actors = []
    category = ['person', 'people', 'officials', 'human', 'men', 'woman', 'organization', "criminals"]
    for each_word in actor_list:
        for cat in category:
            if each_word in model and cat in model:
                similarity_score = model.similarity(each_word, cat)
                if(similarity_score > 0.25):
                    filtered_actors.append(each_word)
    return list(set(filtered_actors))


def get_all_actor_list(doc_):
    all_actors = []
    for tok in doc_:
        if tok.pos_ == "PROPN" and (tok.ent_type_ in NER_List or not tok.ent_type_):
            all_actors.append(tok.text)
        elif tok.pos_ == "PRON":
            if doc_._.coref_chains.resolve(doc_[tok.i]):
                resolved_names = [tok_.text for tok_ in doc_._.coref_chains.resolve(doc_[tok.i]) \
                                  if tok_.pos_ == "PROPN" and (tok_.ent_type_ in NER_List or not tok_.ent_type_)]
                all_actors.extend(resolved_names)
        elif tok.pos_ == "NOUN":
            all_actors.append(tok.text)
    return all_actors
