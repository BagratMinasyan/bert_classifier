from transformers import Trainer, TrainingArguments
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

class BertClassifier(nn.Module):
    def __init__(self, model_name,  n_output_class = 2):
        super(BertClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name) 
        self.hidden_size = self.encoder.config.hidden_size
        self.fc = nn.Linear(self.hidden_size * 3, n_output_class)

    def average_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        output_1 = self.encoder(input_ids=input_ids_1, attention_mask=attention_mask_1)
        embedding_1 = self.average_pool(output_1.last_hidden_state, attention_mask_1)  
        output_2 = self.encoder(input_ids=input_ids_2, attention_mask=attention_mask_2)
        embedding_2 = self.average_pool(output_2.last_hidden_state, attention_mask_2)
        
        diff = embedding_1 - embedding_2
        concatenated = torch.cat([embedding_1, embedding_2, diff], dim=1)

        logits = self.fc(concatenated)  
        return logits