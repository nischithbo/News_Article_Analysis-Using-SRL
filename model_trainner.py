"""
File: model_trainner.py
Author: Nischith Bairannanavara Omprakash
Date: April 23, 2024
Description: File for bert fine-tuning.
"""
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForTokenClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss
import copy
import json

config = None
config_file = "config.json"
with open(config_file, "r") as file:
        config = json.load(file)["training_config"]

max_length_sentence = config["MAX_SENTENCE_LENGTH"]
line_range = eval(config["line_range"])
article_range = eval(config["article_range"])
N_ACTORS = int(config["N_ACTOR"])
save_model = config["save_model_path"]
save_tokenizer = config["save_tokenizer_path"]
data_file_path = config["data_file_path"]
print(save_model, save_tokenizer, data_file_path)
bert_tokenizer = None


class CustomDataset(Dataset):
    def __init__(self, data_list, max_length):
        self.data_list = data_list
        self.max_length = max_length
        self.bert_tokenizer = bert_tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # print(self.max_length)
        tokens = self.data_list[idx][0]
        actor_label = copy.deepcopy(self.data_list[idx][1])
        verb_label = copy.deepcopy(self.data_list[idx][2])
        actor_label.append([0]*N_ACTORS)
        verb_label.append([0]*N_ACTORS)
        actor_label.insert(0, [0]*N_ACTORS)
        verb_label.insert(0, [0]*N_ACTORS)
        # print(tokens)
        token_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens = tokens)
        # print(token_ids)
        token_ids = [self.bert_tokenizer.cls_token_id] + token_ids + [self.bert_tokenizer.sep_token_id]
        attention_mask = [1]*len(token_ids)
        # for i in range(0, len(actor_label)):
        #   actor_label[i] += [0]*N_ACTORS
        #   verb_label[i] += [0]*N_ACTORS

        pad_length = self.max_length - len(token_ids)
        actor_label += [[0]*N_ACTORS] * pad_length
        verb_label += [[0]*N_ACTORS] * pad_length
        attention_mask += [0] * pad_length
        token_ids += [self.bert_tokenizer.pad_token_id]  * pad_length
        # print(actor_label)
        actor_label_tensor = torch.tensor(actor_label, dtype=torch.float32)
        # print('pass')
        verb_label_tensor = torch.tensor(verb_label, dtype=torch.float32)
        input_ids_tensor =  torch.tensor(token_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        return input_ids_tensor, attention_mask_tensor, actor_label_tensor, verb_label_tensor


def collate_fn(batch):
    # print("Max input_id:", max([ids.max() for ids, _, _, _ in batch]))
    # print("Tokenizer vocab size:", bert_tokenizer.vocab_size)
    max_length = max(len(input_ids) for input_ids, _, _, _ in batch)
    # print(max_length)

    padded_input_ids = []
    padded_attention_masks = []
    padded_actor_labels = []
    padded_verb_labels = []

    for input_ids, attention_mask, actor_labels, verb_labels in batch:
        padding_length = max_length - len(input_ids)

        padded_input_ids.append(torch.cat([input_ids, torch.full((padding_length,), bert_tokenizer.pad_token_id, dtype=torch.long)], dim=0))
        padded_attention_masks.append(torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)], dim=0))

        padded_actor_labels.append(torch.cat([actor_labels, torch.zeros(max_length -  len(actor_labels), actor_labels.size(1), dtype=torch.float32)], dim=0))
        padded_verb_labels.append(torch.cat([verb_labels, torch.zeros(max_length -  len(verb_labels), verb_labels.size(1), dtype=torch.float32)], dim=0))

    batch_input_ids = torch.stack(padded_input_ids)
    batch_attention_masks = torch.stack(padded_attention_masks)
    batch_actor_labels = torch.stack(padded_actor_labels)
    batch_verb_labels = torch.stack(padded_verb_labels)

    return batch_input_ids, batch_attention_masks, batch_actor_labels, batch_verb_labels


def read_csv_row_by_row_pandas(filepath):
    final_data_list = []
    data = pd.read_csv(filepath, encoding='cp1252').dropna()
    for index, row in data.iterrows():
        token_list = eval(row['Tokens'])
        actor_labels = eval(row['actor_labels'])
        action_labels = eval(row['action_labels'])
        final_data_list.append([token_list, actor_labels, action_labels])

    return final_data_list[line_range[0]: line_range[1]]


def train(data, train_from_start=True):
    global bert_tokenizer
    if train_from_start:
        bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=N_ACTORS * 2)
    else:
        bert_tokenizer = RobertaTokenizer.from_pretrained(save_tokenizer)
        model = RobertaForTokenClassification.from_pretrained(save_model)

    # Example sentences

    sentences = data
    additional_tokens = []
    for sentence in sentences:
        # Tokenize sentence with spaCy
        spacy_tokens = sentence[0]
        # print(spacy_tokens)
        for tokens in spacy_tokens:
          if tokens not in bert_tokenizer.get_vocab():
            additional_tokens.append(tokens)
    if additional_tokens:
      unique_tokens = list(set(additional_tokens))
      bert_tokenizer.add_tokens(unique_tokens)
      model.resize_token_embeddings(len(bert_tokenizer))
    bert_tokenizer.save_pretrained(save_tokenizer)
    model.save_pretrained(save_model)

    bert_tokenizer = RobertaTokenizer.from_pretrained(save_tokenizer)
    model = RobertaForTokenClassification.from_pretrained(save_model)
    dataset = CustomDataset(data, max_length_sentence+2)  # Assuming max_length of 50 tokens per sentence
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    loss_function = BCEWithLogitsLoss()

    # Ensure the model and loss function are on the correct device
    device = torch.device("cpu")
    model = model.to(device)
    loss_function = loss_function.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    counter = 0
    for epoch in range(6):
        model.train()
        total_loss = 0
        for input_ids_batch, attention_mask_batch, actor_labels_batch, verb_labels_batch in dataloader:
            input_ids_batch = input_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)
            actor_labels_batch = actor_labels_batch.to(device)
            verb_labels_batch = verb_labels_batch.to(device)

            outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
            logits = outputs.logits
            # Calculate loss manually
            actor_loss = loss_function(logits[:, :, :N_ACTORS].view(-1, N_ACTORS), actor_labels_batch.view(-1, N_ACTORS))
            verb_loss = loss_function(logits[:, :, N_ACTORS:].view(-1, N_ACTORS), verb_labels_batch.view(-1, N_ACTORS))
            loss = actor_loss + verb_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if counter % 100 == 1:
                print(counter)
            counter += 1
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(dataloader)}")
    print("Training Complete")
    model.save_pretrained(save_model)


if __name__ == "__main__":
    data = read_csv_row_by_row_pandas(data_file_path)
    train(data)
