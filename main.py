from bert_classifier import BertClassifier
from custom_collator import CustomCollator
from custom_trainer import CustomTrainer
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from datasets import load_dataset

import wandb
wandb.login(key='c1608bf156b31edc293f057bba0684de2f6dfb40')
wandb.init(
    project="BertClassifier",
    config={}
)

dataset = load_dataset('Metric-AI/wikipedia_arlis_c4_news_query_passage_hard_query_negatives')
train_dataset = dataset['train']
val_dataset = dataset['validation'].select(range(10000))

model = BertClassifier('Metric-AI/arm-me5-mlm')
tokenizer = AutoTokenizer.from_pretrained('Metric-AI/arm-me5-mlm')
custom_collator = CustomCollator(tokenizer,512)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    remove_unused_columns=False
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, 
    eval_dataset=val_dataset, 
    data_collator=custom_collator,
)
trainer.train()