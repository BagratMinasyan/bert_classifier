from transformers import Trainer, TrainingArguments
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits = model(
            input_ids_1=inputs['query_input_ids'],
            attention_mask_1=inputs['query_attention_mask'],
            input_ids_2=inputs['passage_input_ids'],
            attention_mask_2=inputs['passage_attention_mask']
        )

        labels = inputs["labels"]
        loss = F.cross_entropy(logits, labels)
        return (loss, logits) if return_outputs else loss