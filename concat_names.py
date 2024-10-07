"""
File: concat_names.py
Author: Nischith Bairannanavara Omprakash
Date: April 23, 2024
Description: Contains method to concat proper nouns in text.
Copyright (c) 2024 Your Company. All rights reserved.
"""


def combine_strings(strings, current_combination="", index=0, combinations=None):
    if combinations is None:
        combinations = []

    # Base case: if we've reached the end of the strings list
    if index == len(strings):
        # Add the current combination to the list of combinations if its length is at least 2
        if len(current_combination) >= 2:
            combinations.append(current_combination)
        return

    # Include the current string in the combination and recursively call with the next index
    combine_strings(strings, current_combination + strings[index], index + 1, combinations)

    # Exclude the current string in the combination and recursively call with the next index
    combine_strings(strings, current_combination, index + 1, combinations)

    return combinations


def concat_pronoun(doc):
    concated_names = []
    processed_sentence = []
    # print(type(doc.sents))
    i = 0
    while i < len(doc):
      if doc[i].ent_type_ in ['ORG', "GPE", "PRODUCT", "NORP", "LOC", "FAC", "LAW"] and doc[i].ent_iob_ == "B":
        name = ""
        new_names = [doc[i].text]
        j = i+1
        while j < len(doc) and doc[j].ent_iob_ == "I":
          new_names.append(doc[j].text)
          j += 1
        while i < j:
          if i == j - 1:
            name += doc[i].text_with_ws
          else:
            name += doc[i].text
          i += 1
        if len(new_names) > 1:
            concated_names.extend(combine_strings(new_names))
        processed_sentence.append(name)
      elif doc[i].ent_type_ == 'PERSON' and doc[i].ent_iob_ == "B":
        name = ""
        new_names = [doc[i].text]
        j = i+1
        while j < len(doc) and doc[j].ent_iob_ == "I":
          new_names.append(doc[j].text)
          j += 1
        while i < j:
          if i == j - 1:
            name += doc[i].text_with_ws
          else:
            name += doc[i].text
          i += 1
        if len(new_names) > 1:
            concated_names.extend(combine_strings(new_names))
        processed_sentence.append(name)
      elif doc[i].ent_type_ == 'EVENT' and doc[i].ent_iob_ == "B":
        name = ""
        new_names = [doc[i].text]
        j = i+1
        while j < len(doc) and doc[j].ent_iob_ == "I":
          new_names.append(doc[j].text)
          j += 1
        while i < j:
          if i == j - 1:
            name += doc[i].text_with_ws
          else:
            name += doc[i].text
          i += 1
        if len(new_names) > 1:
            concated_names.extend(combine_strings(new_names))
        processed_sentence.append(name)
      else:
        processed_sentence.append(doc[i].text_with_ws)
        i += 1



    # print(processed_sentence)
    concat_text  = ""
    for tok in processed_sentence:
        concat_text += tok
    return concat_text, concated_names
