import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from datasets import BertDataset

class BERTModel(nn.Module):
    def __init__(self, model_path):
        super(BERTModel, self).__init__()
        self.path = model_path
        self.model = BertModel.from_pretrained(self.path)
        self.fc = nn.Linear(768, 64)
        self.fc1 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        #self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, inputs):
        _, out = self.model(inputs)
        out = self.dropout(self.relu(self.fc(out)))
        out = self.dropout(self.fc1(out))
        return self.sigmoid(out)