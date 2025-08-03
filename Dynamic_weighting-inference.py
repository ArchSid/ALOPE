import json
import torch
import pandas as pd
from datasets import Dataset
from transformer_heads.output import HeadedModelOutput
from transformer_heads import create_headed_qlora, load_lora_with_heads
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformer_heads.util.helpers import DataCollatorWithPadding, get_model_params
from transformer_heads.config import HeadConfig
import ast  
import os
from torch.utils.data import DataLoader

# original model path
model_path = "<ORIGINAL_MODEL_PATH>"


trained_model_path = '<PATH_TO_TRAINED_MODEL>'

# Test dataset paths
# Eg: test_data_paths = ["data/test.engu.df.short.tsv", "data/test.enhi.df.short.tsv", "data/test.enmr.df.short.tsv"]                      
test_data_paths = [
    "<TEST_DATA_PATH_1>", 
    "<TEST_DATA_PATH_2>", 
    "<TEST_DATA_PATH_3>",
]

model_params = get_model_params(model_path)
model_class = model_params["model_class"]
print(model_params)

from transformer_heads.constants import model_type_map
from transformers import LlamaForCausalLM
# Change the model type map according to the choosen model
model_type_map["meta-llama"] = ("model", LlamaForCausalLM)

# Load head configurations
head_configs_path = f"{trained_model_path}head_configs.json"
with open(head_configs_path, "r") as f:
    head_configs_data = json.load(f)

head_configs = [HeadConfig(**config) for config in head_configs_data.values()]

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = load_lora_with_heads(
    model_class,
    trained_model_path,
    quantization_config,
    device_map={"": torch.cuda.current_device()},
)

language_map = {
    'en': 'English', 'ta': 'Tamil', 'gu': 'Gujarati', 'hi': 'Hindi',
    'mr': 'Marathi', 'te': 'Telugu', 'ne': 'Nepali', 'si': 'Sinhala',
    'et': 'Estonian', 'de': 'German', 'zh': 'Chinese'
}

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def processing_function(examples, source_language_name, target_language_name):
    source_segs = examples['original']
    target_segs = examples['translation']
    prompts = [
        f'''
        Score the following translation from {source_language_name} to {target_language_name} on a continuous scale from 0 to 100, 
        where a score of zero means "no meaning preserved" and a score of one hundred means "perfect meaning and grammar". 
        {source_language_name} source: "{source_seg}" {target_language_name} translation: "{target_seg}" Score: 
        '''

        for source_seg, target_seg in zip(source_segs, target_segs)
    ]

    out = tokenizer(prompts, padding=False, truncation=True)

    mean_values = []
    for x in examples['mean']:
        try:
            mean_values.append(float(x))
        except ValueError:
            try:
                values_list = ast.literal_eval(x)
                mean_values.append(sum(values_list) / len(values_list))
            except (ValueError, SyntaxError):
                mean_values.append(0.0)

    out["mean_regression"] = torch.tensor(mean_values, dtype=torch.float32)
    return out

collator = DataCollatorWithPadding(
    feature_name_to_padding_value={
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": 0,
    }
)

output_dir = '<output_dir_path>'
os.makedirs(output_dir, exist_ok=True)

for test_data_path in test_data_paths:
    language_pair = test_data_path.split('/')[-1].split('.')[1]
    language_pair_formatted = language_pair[:2] + '-' + language_pair[2:]
    source_language_name = language_map.get(language_pair[:2], "Unknown")
    target_language_name = language_map.get(language_pair[2:], "Unknown")
    print(f"Processing: {language_pair_formatted}")

    mt_test_df = Dataset.from_pandas(pd.read_csv(test_data_path, sep='\t'))
    mt_test_df = mt_test_df.map(lambda x: processing_function(x, source_language_name, target_language_name), batched=True)
    mt_test_df.set_format(type="torch", columns=["input_ids", "attention_mask", "mean_regression"])
    mt_test_df = mt_test_df.remove_columns(["index", "original", "translation", "mean", "scores", "z_scores", "z_mean"])

    data_loader = DataLoader(mt_test_df, batch_size=1, collate_fn=collator)
    results = []

    for i, batch in enumerate(data_loader):
        batch = {key: val.to(torch.cuda.current_device()) for key, val in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        
        final_value_tensor = output.preds_by_head['mean_regression'][0, -1, 0]
        final_value = final_value_tensor.item()
        ground_truth = mt_test_df['mean_regression'][i].item()
        results.append((i, ground_truth, final_value))

      
        print(f"Index: {i}, Ground Truth: {ground_truth}, Prediction: {final_value}")

    output_file_path = os.path.join(output_dir, f'{language_pair_formatted}_Llama-3.2-3B_layer_weight_inference_results.csv')
    results_df = pd.DataFrame(results, columns=['index', 'ground_truth', 'prediction'])
    results_df.to_csv(output_file_path, index=False)
    print(f'Results saved to: {output_file_path}')
