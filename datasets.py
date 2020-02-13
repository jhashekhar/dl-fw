import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class BertDataset(Dataset):

    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    # i want to get out a vector of numbers that /
    # can be used as input for the transformer model
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = str(self.texts[idx])
        target = self.targets[idx]
        #tokenizer = BertTokenizer.from_pretrained('vocab.txt')
        inputs = self.tokenizer.encode(text, 
                                       add_special_tokens=True, 
                                       max_length=self.max_len)
        
        # padding - forgot to do the padding threw a size of tensor mismatch error
        padding_len = self.max_len - len(inputs)
        inputs = inputs + ([0] * padding_len)
        # print(torch.tensor(inputs, dtype=torch.long).shape)
        return {'input_ids': torch.tensor(inputs, dtype=torch.long), 
                'targets': torch.tensor(target, dtype=torch.float)}