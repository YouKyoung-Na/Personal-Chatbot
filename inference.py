import math
import re
import random
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel

class Inference():
    def __init__(self):
        # special token
        self.BOS = "</s>" 
        self.EOS = "</s>"
        self.UNK = "<unk>" 
        self.PAD = "<pad>" 
        self.MASK = "<unused0>"
        self.ENTER = "<ENTER>"

        self.Q_TKN = "<usr>"
        self.A_TKN = "<sys>"
        self.SENT = "<unused1>"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", 
                                                                 bos_token=self.BOS, 
                                                                 eos_token=self.EOS, 
                                                                 unk_token=self.UNK, 
                                                                 pad_token=self.PAD, 
                                                                 mask_token=self.MASK)
        self.model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

    def model_load(self, path):
        
        state_dict = torch.load(path, map_location=self.device)

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.`
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        print("[+] Model load complete")

    def inference(self, msg):
        answer = ""
        while True:
            input_ids = torch.LongTensor(self.tokenizer.encode(self.Q_TKN + str(msg) + self.SENT + self.A_TKN + answer)).unsqueeze(dim=0).to(self.device)
            predict = self.model(input_ids)
            predict = predict.logits
            predict = self.tokenizer.convert_ids_to_tokens(torch.argmax(predict, dim=-1).squeeze().cpu().detach().numpy().tolist())[-1]
            
            if (predict == self.EOS) or (predict == self.PAD):
                break
            answer += predict.replace("▁", " ")
        
        return answer.strip()
