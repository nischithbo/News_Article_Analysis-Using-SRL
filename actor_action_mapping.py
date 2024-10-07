"""
File: actor_action_mapping.py
Author: Nischith Bairannanavara Omprakash
Date: April 23, 2024
Description: Contains helper methods to map actor and action words.
"""
import copy

NER_List = ['ORG', "GPE", "PRODUCT", "NORP", "LOC", "FAC", "LAW", 'PERSON', 'EVENT']
invalid_names = [
	"Mr.", "Mrs.", "Ms.", "Mx.", "Dr.", "Prof.", "Rev.", "Fr.", "Sr.", "Br.",
	"Capt.", "Sgt.", "Lt.", "Gen.", "Hon.", "Pres.", "Gov.", "Atty.", "Dean", "Chancellor", "Rector",
	".", ",", "!", "?", ":", ";", "-", "--", "'", "\"", "(", ")", "[", "]", "{", "}",
	"/", "\\", "@", "#", "$", "%", "^", "&", "*", "_", "~", "`", "|", "…"
]


def get_all_actors(doc_, noun_actor_list):
	all_actors = []
	for tok in doc_:
		if tok.pos_ == "PROPN" and (tok.ent_type_ in NER_List or not tok.ent_type_):
			all_actors.append(tok.text)
		elif tok.pos_ == "PRON":
			if doc_._.coref_chains.resolve(doc_[tok.i]):
				resolved_names = [tok_.text for tok_ in doc_._.coref_chains.resolve(doc_[tok.i]) \
				                  if tok_.pos_ == "PROPN" and (tok_.ent_type_ in NER_List or not tok_.ent_type_)]
				all_actors.extend(resolved_names)
		elif tok.text in noun_actor_list:
			all_actors.append(tok.text)
	return all_actors


def map_actor_to_action(all_actors, verb_mapping, doc_concat):
	all_actors = [actor for actor in all_actors if actor not in invalid_names]
	actor_action_final_mapping = {}
	for name in all_actors:
		if name not in actor_action_final_mapping:
			actor_action_final_mapping[name] = [all_actors.count(name), []]

	for verb in verb_mapping:
		verb_text = doc_concat[verb].text
		actors = set(verb_mapping[verb][0] + verb_mapping[verb][1])
		for actor in actors:
			if actor in all_actors:
				actor_action_final_mapping.get(actor)[1].append(verb_text)
	return actor_action_final_mapping


def merge_repeated_names(actor_action_final_mapping, concatenated_names):
	actor_action_mapping = copy.deepcopy(actor_action_final_mapping)
	all_keys = actor_action_final_mapping.keys()
	for each_actor in all_keys:
		pop = False
		for key in actor_action_mapping:
			if key != each_actor and each_actor in key:
				if key in concatenated_names and each_actor in concatenated_names:
					actor_action_mapping[key][0] += actor_action_mapping[each_actor][0]
					actor_action_mapping[key][1].extend(actor_action_mapping[each_actor][1])
					pop = True
					break
		if pop:
			actor_action_mapping.pop(each_actor)
	return actor_action_mapping


def merge_bert_spacy_mapping(spacy_mapping, bert_mapping):
	for key in spacy_mapping:
		if key in bert_mapping:
			actions = bert_mapping[key][1]
			for action in actions:
				spacy_count = spacy_mapping[key][1].count(action)
				bert_count = spacy_mapping[key][1].count(action)
				if bert_count > spacy_count:
					diff = spacy_count - bert_count
					spacy_mapping[key][1].extend([action]*diff)

	return spacy_mapping



