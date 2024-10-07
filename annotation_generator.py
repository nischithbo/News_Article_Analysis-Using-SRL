"""
File: annotation_generator.py
Author: Nischith Bairannanavara Omprakash
Date: April 23, 2024
Description: Contains method to generate annotations.
"""


NER_List = ['ORG', "GPE", "PRODUCT", "NORP", "LOC", "FAC", "LAW", 'PERSON', 'EVENT']
N_actors = 20


def resolve_pronoun_for_tokens(tok, doc_):
    output = []
    nouns = doc_._.coref_chains.resolve(tok)
    if not nouns:
        return [tok.text]
    if len(nouns) == 1:
        return [nouns[0].text]
    else:
        for j in range(0, len(nouns)):
           if not output:
               output.append(nouns[j].text)
           elif j == len(nouns) - 1:
                output.append("and")
                output.append(nouns[j].text)
           else:
                output.append(",")
                output.append(nouns[j].text)
    return output


def perform_annotations(doc, verb_actor_mapping, noun_actor_list):
    final_annotation = []

    for sentence in doc.sents:
        # print(type(sentence.text))
        # print(sentence)
        new_tokens_mapping = []
        names = []
        if "Â©" in sentence.text or "(Bloomberg)" in sentence.text or not sentence.text:
            continue
        for tok in sentence:
            # print(tok, tok.pos_)
            if tok.pos_ == "PRON":
                resolved_names = resolve_pronoun_for_tokens(tok, doc)
                for name in resolved_names:
                    new_tokens_mapping.append([name, -2])
                if doc._.coref_chains.resolve(tok):

                    for name in doc._.coref_chains.resolve(tok):
                        if name.text not in names and name.ent_type_ in NER_List:
                            names.append(name.text)
            elif  tok.pos_ == "PROPN" and (tok.ent_type_ in NER_List or not tok.ent_type_):
                new_tokens_mapping.append([tok.text, -2])
                if tok.text not in names:
                    names.append(tok.text)
                # names.append(tok.text)
            elif tok.text in noun_actor_list:
                new_tokens_mapping.append([tok.text, -2])
                if tok.text not in names:
                    names.append(tok.text)
            else:
                 if tok.pos_ == "VERB":
                    new_tokens_mapping.append([tok.text, tok.i])
                 else:
                    new_tokens_mapping.append([tok.text, -1])
        # print(new_tokens_mapping)
        # print(names)
        final_sentence_token = []
        final_sentence_annotation_actor = []
        final_sentence_annotation_action = []
        for token in new_tokens_mapping:
            actor_annotation = [0]*N_actors
            action_annotation = [0]*N_actors
            if token[1] == -2:
                if token[0] in names:
                    index  = names.index(token[0])
                    if index < N_actors:
                        actor_annotation[index] = 1
            elif token[1] > -1:
                actors = verb_actor_mapping.get(token[1], [[], []])
                involved_actors = actors[0] + actors[1]
                for act in involved_actors:
                    if act in names:
                        index = names.index(act)
                        if index < N_actors:
                            action_annotation[index] = 1
            else:
                try:
                    eval(str([token]))
                except Exception as err:
                    continue

            final_sentence_token.append(token[0])
            final_sentence_annotation_actor.append(actor_annotation)
            final_sentence_annotation_action.append(action_annotation)
        try:
            eval(str(final_sentence_token))

        except Exception as err:
                    continue
        final_annotation.append([str(sentence), final_sentence_token,  final_sentence_annotation_actor, final_sentence_annotation_action])
    return final_annotation
