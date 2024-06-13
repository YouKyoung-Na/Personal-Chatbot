import math
import numpy as np
import pandas as pd
import random
import re
import tqdm
import os
import argparse

import torch
import urllib.request
from torch.utils.data import DataLoader, Dataset

from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument("--traindata", type=str, help="train data")
parser.add_argument("--pretrain", type=str, help="pretrain weight")
parser.add_argument("--epoch", type=int, help="epoch")
args = parser.parse_args()

Chatbot_Data = pd.read_csv("args.traindata")
Chatbot_Data = Chatbot_Data.dropna(axis=0, how="any")

BOS = "</s>"
EOS = "</s>"
UNK = "<unk>"
PAD = "<pad>"
MASK = "<unused0>"
ENTER = "<ENTER>"

Q_TKN = "<usr>"
A_TKN = "<sys>"
SENT = "<unused1>"


tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token=BOS, eos_token=EOS, unk_token=UNK, pad_token=PAD, mask_token=MASK)


class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=60):
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn["Q"]
        a = turn["A"]

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        if q_len > self.max_len:
            a_len = self.max_len - q_len 

            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len / 2)):] 
                q_len = len(q_toked)
                a_len = self.max_len - q_len

            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len

            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len / 2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len

            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        labels = [self.mask,] * q_len + a_toked[1:]
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)

        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)

        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        return (token_ids, np.array(mask), labels_ids)
    
def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]

    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

train_set = ChatbotDataset(Chatbot_Data, max_len=60)
train_dataloader = DataLoader(train_set, batch_size=128, num_workers=2, shuffle=True, collate_fn=collate_batch)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

if args.pretrain:
    state_dict = torch.load("args.pretrain")

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

learning_rate = 3e-5
criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
Sneg = -1e18

for epoch in range(args.epoch):
    for token_ids, mask, label in tqdm.tqdm(train_dataloader):
        token_ids, mask, label = token_ids.to(device), mask.to(device), label.to(device)
        out = model(token_ids)
        out = out.logits

        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))

        loss = criterion(mask_out.transpose(2, 1), label)
        avg_loss = loss.sum() / mask.sum()

        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

        
    torch.save(model.state_dict(), f"./model_weight/train/fintuning_{epoch}.pt")

    print("LOSS", avg_loss)

