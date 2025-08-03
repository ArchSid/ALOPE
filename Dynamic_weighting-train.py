from transformer_heads import create_headed_qlora, load_lora_with_heads
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    Trainer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorWithPadding,
)
from transformer_heads.util.helpers import get_model_params
from transformers import AutoConfig
from peft import LoraConfig
from transformer_heads.config import HeadConfig
from transformer_heads.util.model import print_trainable_parameters
from transformer_heads.util.evaluate import evaluate_head_wise
import torch
import pandas as pd
import os
import csv
from torch.utils.data import DataLoader
import torch.nn as nn


import torch.nn.functional as F

class WeightedEmbeddingCombiner(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super().__init__()
        
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        self.hidden_size = hidden_size
        
    def forward(self, layer_embeddings):
        weights = F.softmax(self.layer_weights, dim=0)
        
        weighted_sum = torch.zeros_like(layer_embeddings[0])
        for i, embedding in enumerate(layer_embeddings):
            weighted_sum += weights[i] * embedding
            
        return weighted_sum

# Change your selected model
model_path = "meta-llama/Llama-3.2-3B-Instruct"



train_batch_size = 8
eval_batch_size = 8
train_epochs = 1
eval_epochs = 1
logging_steps = 20

model_params = get_model_params(model_path)
model_class = model_params["model_class"]
hidden_size = model_params["hidden_size"]
vocab_size = model_params["vocab_size"]
print(model_class)

from transformer_heads.constants import model_type_map, loss_fct_map
from transformers import GemmaModel, LlamaForCausalLM, MistralModel, AutoModelForCausalLM , MistralForCausalLM,MixtralForCausalLM,MixtralForCausalLM
# Change the model type map according to the choosen model
model_type_map["meta-llama"] = ("model", LlamaForCausalLM)

head_configs = []

'''Specify the layers for which you want to add auxilary heads to recieve the embeddings from
Eg: for layers 17 to 24'''
for layer_idx in range(17, 25):
    head_configs.append(
        HeadConfig(
            name=f"layer_{layer_idx}_embed",
            layer_hook=-layer_idx,
            in_size=hidden_size,
            output_activation="linear",
            is_causal_lm=False,
            pred_for_sequence=True,
            loss_fct=None,  
            num_outputs=hidden_size,  
            is_regression=False,
            loss_weight=0.0,  
        )
    )


# Add the final aggregation head for regression
head_configs.append(
    HeadConfig(
        name="mean_regression",
        layer_hook=-1,  
        in_size=hidden_size,
        output_activation="linear",
        is_causal_lm=False,
        pred_for_sequence=True,
        loss_fct="mse",
        num_outputs=1,  
        is_regression=True,
        loss_weight=0.002,
    )
)

# Define the training files path
# eg: train_files = {"English-Tamil": "data/train.en-ta.df.short.tsv"}
train_files = {
    "English-Tamil": "<TRAIN_DATA_PATH>",
    "English-Telugu": "<TRAIN_DATA_PATH>",
    "English-Hindi": "<TRAIN_DATA_PATH>",
    "English-Gujarati": "<TRAIN_DATA_PATH>",
    "English-Marathi": "<TRAIN_DATA_PATH>",
    "Estonian-English": "<TRAIN_DATA_PATH>",
    "Nepali-English": "<TRAIN_DATA_PATH>",
    "Sinhala-English": "<TRAIN_DATA_PATH>",
}

# Load datasets
def load_and_prepare_datasets(files):
    datasets = []
    for lang_pair, file_path in files.items():
        source_lang, target_lang = lang_pair.split('-')
        df = pd.read_csv(file_path, sep='\t', quoting=csv.QUOTE_NONE)
        df['source_lang'] = source_lang
        df['target_lang'] = target_lang
        datasets.append(Dataset.from_pandas(df))
    return datasets

train_datasets = load_and_prepare_datasets(train_files)

combined_train_dataset = concatenate_datasets(train_datasets)

tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def processing_function(examples):
    source_segs = examples['original']
    target_segs = examples['translation']
    source_langs = examples['source_lang']
    target_langs = examples['target_lang']

    prompts = [
        f'Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100, where a score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar". {source_lang} source: "{source_seg}" {target_lang} translation: "{target_seg}" Score: '
        for source_seg, target_seg, source_lang, target_lang in zip(source_segs, target_segs, source_langs, target_langs)
    ]
    out = tokenizer(prompts, padding=False, truncation=True)

    out["mean_regression"] = examples['mean']
    return out

combined_train_dataset = combined_train_dataset.map(processing_function, batched=True)

combined_train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "mean_regression"],
)



print(f"Number of data points used for training: {len(combined_train_dataset)}")
print(combined_train_dataset[0])



quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = create_headed_qlora(
    base_model_class=model_class,
    model_name=model_path,
    quantization_config=quantization_config,
    lora_config=lora_config,
    head_configs=head_configs,
    fully_trained_heads=True,
    device_map={"": torch.cuda.current_device()},
    gradient_checkpointing=True,
    trust_remote_code=True
)

original_forward = model.forward

# dynamic weighting 
'''here 8 is the number of layers, please change it according to the number of layers you are using'''
model.embedding_combiner = WeightedEmbeddingCombiner(8, hidden_size)

def custom_forward(self, input_ids=None, attention_mask=None, **kwargs):
    outputs = original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    layer_embeddings = []

    # Collect embeddings from all layer heads from 17 to 24
    for layer_idx in range(17, 25):
        layer_key = f'layer_{layer_idx}_embed'
        if layer_key in outputs["preds_by_head"]:
            layer_embeddings.append(outputs["preds_by_head"][layer_key])


    # Only proceed if we have collected all required embeddings
    '''here 8 is the number of layers, please change it according to the number of layers you are using'''
    if len(layer_embeddings) == 8:
        combined_embedding = self.embedding_combiner(layer_embeddings)
                
        outputs['mean_regression'] = self.heads['mean_regression'](combined_embedding)
    return outputs

model.forward = custom_forward.__get__(model)


combined_train_dataset = combined_train_dataset.remove_columns(["original", "translation", "mean", "scores", "z_scores", "z_mean", "source_lang", "target_lang"])

collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=None,
    return_tensors="pt"
)

def custom_collator(features):
    batch = collator(features)
    batch['mean_regression'] = torch.tensor([f['mean_regression'] for f in features], dtype=torch.float)
    return batch

data_loader = DataLoader(combined_train_dataset, batch_size=1, collate_fn=custom_collator)
for batch in data_loader:
    print(batch)
    break

args = TrainingArguments(
    output_dir="./result",
    learning_rate=0.0002,
    num_train_epochs=train_epochs,
    logging_steps=logging_steps,
    do_eval=False,
    remove_unused_columns=False,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    lr_scheduler_type="constant", 
    ddp_find_unused_parameters=False,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)

trainer = Trainer(
    model,
    args=args,
    train_dataset=combined_train_dataset,
    data_collator=custom_collator,
)


trainer.train()

log_file_path = "training_log.csv"

os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

training_logs = trainer.state.log_history

headers = ["epoch", "loss", "grad_norm", "learning_rate"]

with open(log_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(headers)
    
    for log in training_logs:
        if "loss" in log:  
            writer.writerow([
                log.get("epoch", "N/A"),  
                log.get("loss", "N/A"),   
                log.get("grad_norm", "N/A"),  
                log.get("learning_rate", "N/A")  
            ])

print(f"Training logs saved at: {log_file_path}")


trainer.save_model("<SAVE_MODEL_PATH>")