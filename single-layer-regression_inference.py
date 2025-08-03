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

# Original model path
model_path = "<ORIGINAL_MODEL_PATH>"


# Trained model path for inferencing
trained_model_path = '<PATH_TO_TRAINED_MODEL>'

# Test dataset path
test_data_path = '<TEST_DATA_PATH>'

# Extract the layer identifier from the trained model path eg: Our trained models are named as Llama-3.1-8B_layer_-7. 
# You can change this according to your model names
layer_identifier = trained_model_path.rstrip('/').split('_')[-1]
print(f"layer::: {layer_identifier}")

model_params = get_model_params(model_path)
model_class = model_params["model_class"]
vocab_size = model_params["vocab_size"]
print(model_params)



from transformer_heads.constants import model_type_map, loss_fct_map
import torch.nn as nn
from transformers import LlamaForCausalLM

# Change the model type map according to the choosen model
model_type_map["meta-llama"] = ("model", LlamaForCausalLM)


# Load head configurations
head_configs_path = f"{trained_model_path}head_configs.json"
with open(head_configs_path, "r") as f:
    head_configs_data = json.load(f)

head_configs = [
    HeadConfig(**config) for config in head_configs_data.values()
]


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

# Extract language pair from the test data path
language_pair = test_data_path.split('/')[-1].split('.')[1]  # eg:Extract 'eten' from the datafile named as 'test.eten.df.short.tsv'
language_pair_formatted = language_pair[:2] + '-' + language_pair[2:]  # eg: Convert 'eten' to 'et-en'

# Language code to language name mapping
language_map = {
    'en': 'English',
    'ta': 'Tamil',
    'gu': 'Gujarati',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'te': 'Telugu',
    'ne': 'Nepali',
    'si': 'Sinhala',
    'et': 'Estonian',
    'de': 'German',
    'zh': 'Chinese'
}

# Determine source and target language names
source_language_name = language_map.get(language_pair[:2], "Unknown")
target_language_name = language_map.get(language_pair[2:], "Unknown")

# Print the language mapping
print(f"Source language: {source_language_name}, Target language: {target_language_name}")

# Load the test dataset
mt_test_df = Dataset.from_pandas(pd.read_csv(test_data_path, sep='\t'))

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

first_prompt_printed = False

def processing_function(examples):
    global first_prompt_printed

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

    if not first_prompt_printed and prompts:
        print("First prompt:", prompts[0])  
        first_prompt_printed = True

    out = tokenizer(prompts, padding=False, truncation=True) 


    
    mean_values = []
    for x in examples['mean']:
        try:
            
            mean_values.append(float(x))
        except ValueError:
            
            try:
                values_list = ast.literal_eval(x)  
                mean_values.append(sum(values_list) / len(values_list))
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing mean value: {x}, Error: {e}")
                mean_values.append(0.0)  

    out["mean_regression"] = torch.tensor(mean_values, dtype=torch.float32)  


    return out


mt_test_df = mt_test_df.map(processing_function, batched=True)

mt_test_df.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"] + [x.name for x in head_configs],
)

mt_test_df = mt_test_df.remove_columns(["index", "original", "translation", "mean", "scores", "z_scores", "z_mean"])


class DebugDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        
        batch = super().__call__(features)
        
        return batch


collator = DebugDataCollatorWithPadding(
    feature_name_to_padding_value={
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": 0,
    }
)


from torch.utils.data import DataLoader
total_error = 0

results = []


data_loader = DataLoader(mt_test_df, batch_size=1, collate_fn=collator)

for i, batch in enumerate(data_loader):
    batch = {key: val.to(torch.cuda.current_device()) for key, val in batch.items()}

    
    with torch.no_grad():
        output = model(**batch)

    output = model(**batch)
    out = {}
    for head_name in output.preds_by_head:
        logits = output.preds_by_head[head_name]
        pred_logits = logits[0, -1, :] 
        best_n = torch.topk(pred_logits, 1) 
        out[head_name] = [tokenizer.decode(i) for i in best_n.indices] 

    
    mean_regression_tensor = output.preds_by_head['mean_regression']

    
    final_value_tensor = mean_regression_tensor[0, -1, 0]

    
    final_value = final_value_tensor.item()
    ground_truth = mt_test_df['mean_regression'][i].item()  
    print(f'Ground Truth {ground_truth} & Prediction Value {final_value}')
    
    
    results.append((i, ground_truth, final_value))
    
    
    total_error += (ground_truth - final_value)**2

# Save the output file path using the layer identifier and language pair - you can change this accordingly
output_dir = f'<OUTPUT_MODEL_PATH>/<MODEL_NAME>_layer_{layer_identifier}/'
output_file_path = os.path.join(output_dir, f'{language_pair_formatted}_<MODEL_NAME>_layer_{layer_identifier}_inference_results.csv')


os.makedirs(output_dir, exist_ok=True)  

# Save results
results_df = pd.DataFrame(results, columns=['index', 'ground_truth', 'prediction'])
results_df.to_csv(output_file_path, index=False)

print(f'Results saved to: {output_file_path}')

