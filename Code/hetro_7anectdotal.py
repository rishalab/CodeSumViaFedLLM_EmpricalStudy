#cosine scheduler coz rouge-l and meteor is better in cosine than cnst. The TRN param is coppied as its populating in the csv.
#Create a lora adapter, init it according to QLora paper. eval it acc to Qlora paper. all compile both for train and inference is removed.
import torch
import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache() # PyTorch thing
gc.collect()

LLM_cfgfile = "LLMbase_config.py"
# Import variables from the configuration file
exec(open(LLM_cfgfile).read())
input_cfgfile = "FedCS_LLMinput.py"
# Import variables from the configuration file
exec(open(input_cfgfile).read())
print("Read the config files")

this_BEST_round = 12

import subprocess
packages_to_install = [
    "nltk",
    "rouge",
    "transformers==4.31.0",
    "datasets==2.13.0",
    "peft==0.4.0",
    "accelerate==0.21.0",
    "bitsandbytes==0.40.2",
    "trl==0.4.7",
    "safetensors>=0.3.1",
    "ipywidgets==7.7.1",
    "huggingface_hub",
    "python-dotenv",
    "scipy",
    "pandas",
    "sentencepiece"
]

# Install the packages using subprocess
# for package in packages_to_install:
#    subprocess.run(["pip", "install", package])
print("Installations completed")

output_dir = "NonFed/"
# test_dataset_dir = "../code_docstring_corpus_data_test/"
from datasets import load_dataset
from random import randrange
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, LlamaTokenizer, LlamaForCausalLM, Trainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, get_peft_model_state_dict, AutoPeftModelForCausalLM, prepare_model_for_int8_training
from trl import SFTTrainer

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import os
import random
from datasets import Dataset
import csv
import json
from datasets import load_from_disk
from collections import OrderedDict
import datetime
import nltk
# nltk.download('wordnet')
import sys
from io import StringIO
print("Imports completed")

import pandas as pd
import json
from huggingface_hub import login
from dotenv import load_dotenv
import os
# credentials_filename = "../credentials.txt"
# # Read the token from the file
# token = None
# with open(credentials_filename, "r") as file:
#     for line in file:
#         if "HF_HUB_TOKEN" in line:
#             token = line.split("=")[1].strip()
#             break
# # Check if a token was found
# if token is None:
#     raise ValueError("HF_HUB_TOKEN not found in credentials.txt")
# # Login to the Hugging Face Hub
# login(token=token)
# print("Logged into huggingface")

#print versions of Torch and cuda if available. 
def print_torch_cuda_info():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
print_torch_cuda_info()

#Create Train data
import os
from datasets import Dataset
#Read the datafiles
import pandas as pd
import re

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
# tokenizer.pad_token = 0
# tokenizer.padding_side = "left"
print("tokenizer defined")

import datetime
import torch
from peft import PeftModel
import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
#Evaluate model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE
train_dataset = load_from_disk(train_dataset_dir)

# #Cases description
# weight_matrices = ''.join([module[0] for module in LORA_TARGET_MODULES])
# adapter_dir = output_dir+'Ada_W'+weight_matrices+'_r'+str(lora_r)+'/'
def format_instruction(sample):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{"generate summary for the below java function"}
### Input:
{sample["code"]}
### Response:
{sample["summary"]}"""

def generate_docstring(merged_model, sample, tokenizer):
    prompt =  f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {"generate summary for the below java function"}
    ### Input:
    {sample["code"]}
    ### Response:
    """
        # Tokenize the prompt and generate the docstring
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.5)
    # Extract and format the generated docstring
    generated_docstring = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    # Return the generated and ground truth docstrings
    return sample["code"], generated_docstring, sample['summary']

# generated, ground_truth = generate_docstring(sample, tokenizer)
def save_docstring_to_csv(merged_model, test_dataset, tokenizer, file_path):
    # Initialize lists to store data
    all_functions = []
    all_generated = []
    all_ground_truth = []

    # Loop over the entire test dataset to generate docstrings
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        this_function, generated, ground_truth = generate_docstring(merged_model, sample, tokenizer)
        all_functions.append(this_function)
        all_generated.append(generated)
        all_ground_truth.append(ground_truth)

    # Health check of generated data:
    for idx, text in enumerate(all_generated):
        if not text.strip():  # If text is empty or just whitespace
            print(f"Empty string found at index {idx}")
            all_generated[idx] = "NO_OUTPUT"
    for idx, text in enumerate(all_ground_truth):
        if not text.strip():  # If text is empty or just whitespace
            print(f"Empty reference found at index {idx}")
            all_ground_truth[idx] = "NO_INPUT"
            all_generated[idx] = "SO_NO_OUTPUT"

    if not os.path.exists(file_path):
        # Create the CSV file and write the headers
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Function", "Generated", "Ground Truth"])

    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(all_functions)):
            writer.writerow([all_functions[i], all_generated[i], all_ground_truth[i]])
    return "Generated"

def merge_lora2base(adapter_dir):
    new_model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, low_cpu_mem_usage=True,
    return_dict=True, torch_dtype=torch.float16, device_map=device_map,) #This is the adapters reloaded.
    # Merge LoRA and base model
    merged_model = new_model.merge_and_unload()
    return merged_model

def this_r_W_experiment(adapter_dir, weight_matrices, LORA_TARGET_MODULES, lora_r, output_file):
    merged_model = merge_lora2base(adapter_dir)
    test_dataset = load_from_disk(test_dataset_dir)
    status = save_docstring_to_csv(merged_model, test_dataset, tokenizer, output_file)
    del merged_model
    if status!="Generated":
        print("ERROR: Examples arent generated", "please check.")
    else:
        print("Examples are generated for Central model for r=", lora_r,"and W=", weight_matrices )

# # Central Model:
# print("\n\n----------------------------------------------------------------------")
# print("Beginning Central experiment for W=", LORA_TARGET_MODULES, "with r=", lora_r)
# print("EXP Start time:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
# weight_matrices = ''.join([module[0] for module in LORA_TARGET_MODULES])
# adapter_dir = output_dir+'Central/TrD'+str(num_train_rows)+'Ada_W'+weight_matrices+'_r'+str(lora_r)+'/'
# this_exp = "Central"
# this_r_W_experiment(adapter_dir, weight_matrices, LORA_TARGET_MODULES, lora_r, this_exp+"_anecdotal.csv")
# print("SEeeeeee!! "+this_exp+" experiment examples generated")
# print("EXP End time:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

# # Round 0 (Pre-trained):
# this_exp = "PreTrained"
# print("Beginning"+this_exp+" experiment")
# print("EXP Start time:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
# output_dir = 'round0/server/'
# adapter_dir = output_dir+'Ada/'
# weight_matrices = ''.join([module[0] for module in LORA_TARGET_MODULES])
# this_r_W_experiment(adapter_dir, weight_matrices, LORA_TARGET_MODULES, lora_r, this_exp+"_anecdotal.csv")
# print("SEeeeeee!! "+this_exp+" experiment examples generated")
# print("EXP End time:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

# import gc
# gc.collect()
# gc.collect()
# torch.cuda.empty_cache() # PyTorch thing
# gc.collect()
import os  
import shutil  

#generate previous two round merged_models.
this_round = 0
def del_base_model(this_round):
    #check if the "round_"+prev_round+"/server/merged_model" directory exists
    directory_path = "round"+str(this_round)+"/server/merged_model_Impr"
    # Check if the directory exists
    if os.path.exists(directory_path):
        print(f"The directory '{directory_path}' exists.")
        shutil.rmtree(directory_path)
        print(f"The directory '{directory_path}' deleted.")

def check_base_model(this_round):
    #check if the "round_"+prev_round+"/server/merged_model" directory exists
    directory_path = "round"+str(this_round)+"/server/merged_model_Impr"
    adapter_dir = "round"+str(this_round)+"/server/Ada_Impr"
    # Check if the directory exists
    if os.path.exists(directory_path):
        print(f"The directory '{directory_path}' exists.")
    else:
        print(f"The directory '{directory_path}' does not exist.")
        model = merge_lora2base(adapter_dir)
        # Save the model to the specified directory
        if this_round >=2:
            del_base_model(this_round-2)
            print("Deleted ", this_round, "merged model")
        model.save_pretrained(directory_path)
        print(f"Model has been saved to '{directory_path}'.")

#Round BEST:
this_exp = "FedBEST"
for this_round in range(0, this_BEST_round):
    check_base_model(this_round)
print("Base for Fed BEST @", this_BEST_round, "prepared")
print("Beginning"+this_exp+" experiment")
print("EXP Start time:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
output_dir = 'round'+str(this_BEST_round)+'/server/'
adapter_dir = output_dir+'Ada_Impr/'
weight_matrices = ''.join([module[0] for module in LORA_TARGET_MODULES])
this_r_W_experiment(adapter_dir, weight_matrices, LORA_TARGET_MODULES, lora_r, this_exp+"_anecdotal.csv")
print("SEeeeeee!! "+this_exp+" experiment examples generated")
print("EXP End time:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

del_base_model(this_BEST_round-2)
del_base_model(this_BEST_round-1)
del_base_model(this_BEST_round)
print("Deleted the BEST-2, BEST-1 and BEST merged_model")

import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache() # PyTorch thing
gc.collect()