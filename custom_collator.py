from transformers import Trainer, TrainingArguments
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

class CustomCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        queries = [item['query'] for item in batch]
        passages = [item['passage'] for item in batch]
        labels = [item['label'] for item in batch]

        query_encoding = self.tokenizer(
            queries,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        passage_encoding = self.tokenizer(
            passages,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'query_input_ids': query_encoding['input_ids'],
            'query_attention_mask': query_encoding['attention_mask'],
            'passage_input_ids': passage_encoding['input_ids'],
            'passage_attention_mask': passage_encoding['attention_mask'],
            'labels': torch.tensor(labels)
        }