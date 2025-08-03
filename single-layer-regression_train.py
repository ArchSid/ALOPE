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



# Original model path
model_path = "<ORIGINAL_MODEL_PATH>"

# Define according to your resource availability
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
import torch.nn as nn
from transformers import LlamaForCausalLM


# Change the model type map according to the choosen model
model_type_map["meta-llama"] = ("model", LlamaForCausalLM)


# Define the head configurations
head_configs = [
    HeadConfig(
        name="mean_regression",
        layer_hook=-1, # layer -1 is the last transformer block.
        in_size=hidden_size,
        output_activation="linear", 
        is_causal_lm=False,
        pred_for_sequence=True,
        loss_fct="mse",
        num_outputs=1,  # Single value output
        is_regression=True,
        loss_weight=0.002,
    ),
]


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

# Combine datasets
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
    columns=["input_ids", "attention_mask"] + [x.name for x in head_configs],
)



print(f"Number of data points used for training: {len(combined_train_dataset)}")
print(combined_train_dataset[0])


# Define the quantization 
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
# Define the LORA configurations
lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

#define the model with the heads
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

# change the values accordingly
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

trainer.save_model("<SAVE_MODEL_PATH>")

