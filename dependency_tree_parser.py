"""
File: dependency_tree_parser.py
Author: Nischith Bairannanavara Omprakash
Date: April 23, 2024,
Description: Contains all methods needed to create action-actor mapping by parsing dependency tree
"""


NER_List = ['ORG', "GPE", "PRODUCT", "NORP", "LOC", "FAC", "LAW", 'PERSON', 'EVENT']


def bfs_traversal(root):
    """Takes root and creates a list of nodes in the order of appearance in tree
    """
    result = []
    queue = [root]

    while queue:
        node = queue.pop(0)
        if node.pos_ == "VERB":
            result.append(node)
        queue.extend(node.children)

    return result


# Constants
subject_tags = [
    "nsubj",          # Nominal subject
    "nsubjpass",      # Nominal subject passive
    # "csubj",          # Clausal subject
    # "expl",           # Expletive subject
    "nsubj:xsubj"     # Controlled nominal subject
]

# Object Tags
object_tags = [
    "dobj",           # Direct object
    # "iobj",           # Indirect object
    # "obj",            # Object
    # "objpass",        # Object passive
    # "csubj",          # Clausal subject
    # "ccomp",
    "pobj"
]

linkers = [
    "ADP"
]

dependency_linkers = [
    "agent"
]


def resolve_pronoun(index, doc_):
    """ Resolves pronoun to proper nouns
    """
    if doc_._.coref_chains.resolve(doc_[index]):
        # print(f"Debug {doc[index].text}: {[tok_.text for tok_ in doc._.coref_chains.resolve(doc[index])]}")
        return [tok_.text for tok_ in doc_._.coref_chains.resolve(doc_[index]) if tok_.pos_ == "PROPN" and (tok_.ent_type_ in NER_List or not tok_.ent_type_)]
    else:
        return []


def check_nsub_exist(token):
    """Checks if given token/node has any children that has subject tags"""
    for child in token.children:
        if child.dep_ in subject_tags:
            return True
    return False


def extract_objects(token, doc_, noun_actor_list):
    """Extracts actors from the token"""
    obj_list = []
    if token.pos_ == "PROPN" and (token.ent_type_ in NER_List or not token.ent_type_):
        obj_list.append(token.text)
    elif token.pos_ == "PRON":
        obj_list.extend(resolve_pronoun(token.i, doc_))
    elif token.text in noun_actor_list:
            obj_list.extend([token.text])
    process_node = []
    processed_node = []
    for child_ in token.children:
        process_node.append(child_)

    while process_node:
        node = process_node.pop(0)
        for each_child in node.children:
            if (each_child.pos_ in ["PROPN", "PRON"] or each_child.dep_ in ["conj", "nmod", "agent"] or each_child.pos_ in linkers or each_child.dep_ in subject_tags or
                each_child.dep_ in object_tags ) and each_child.i not in processed_node:
                process_node.append(each_child)
            elif each_child.text in noun_actor_list:
                process_node.append(each_child)

        if node.pos_ == "PROPN" and (node.ent_type_ in NER_List or not node.ent_type_):
            obj_list.append(node.text)
        elif node.pos_ == "PRON" and (node.ent_type_ in NER_List or not node.ent_type_):
            obj_list.extend(resolve_pronoun(node.i, doc_))

        elif node.text in noun_actor_list:
            obj_list.extend([node.text])

        processed_node.append(node.i)

    return list(obj_list)


def map_verb_actor(doc, noun_actor_list):
    """Main method to create action-actor mapping"""
    root = [token for token in doc if token.head == token]

    verb_list = []

    for r in root:
        verb_list.extend(bfs_traversal(r))

    verb_actor_mapping = {}
    for token in doc:
        if token.pos_ == "VERB":

            verb_actor_mapping[token.i] = [[], []]

    for verb_tok in verb_list:
        # print(verb_tok.text, [tok.text for tok in verb_tok.ancestors])
        nsubj = False
        sub_obj = []
        dir_obj = []
        # print(verb_tok.text, [c.text for c in verb_tok.children])
        for child in verb_tok.children:
            if child.pos_ in linkers or child.dep_ in dependency_linkers:
                for second_level_child in child.children:
                    if second_level_child.dep_ in subject_tags:
                        sub_obj.extend(extract_objects(child, doc, noun_actor_list))
                        nsubj = True
                    if second_level_child.dep_ in object_tags:
                        dir_obj.extend(extract_objects(child, doc, noun_actor_list))

            if child.dep_ in subject_tags:
                sub_obj.extend(extract_objects(child, doc, noun_actor_list))
                nsubj = True
            if child.dep_ in object_tags:
                dir_obj.extend(extract_objects(child, doc, noun_actor_list))

        if not nsubj:
            for ancestor in verb_tok.ancestors:
                if ancestor.pos_ == "VERB":
                    # print(verb_tok.text, ancestor.text)
                    sub_obj.extend(verb_actor_mapping[ancestor.i][0])
                    if check_nsub_exist(ancestor):
                        break
        verb_actor_mapping[verb_tok.i][0].extend(set(sub_obj))
        verb_actor_mapping[verb_tok.i][1].extend(set(dir_obj))

    return verb_actor_mapping
