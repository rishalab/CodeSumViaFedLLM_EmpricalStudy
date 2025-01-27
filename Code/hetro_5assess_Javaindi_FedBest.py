# Check if base exists this_round-1

# if not create it

#For this_round:

#   each client:

#       create merged_model

        # evaluate

        # enter in excel sheet

        # delete merged_model

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



import shutil

from random import randrange

import torch

import transformers

from transformers import AutoModel,AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, LlamaTokenizer, LlamaForCausalLM, Trainer

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

nltk.download('wordnet')

import sys

from io import StringIO

print("Imports completed")



import pandas as pd

import json

from huggingface_hub import login

from dotenv import load_dotenv

import os

credentials_filename = "../credentials.txt"

# Read the token from the file

token = None

with open(credentials_filename, "r") as file:

    for line in file:

        if "HF_HUB_TOKEN" in line:

            token = line.split("=")[1].strip()

            break

# Check if a token was found

if token is None:

    raise ValueError("HF_HUB_TOKEN not found in credentials.txt")

# Login to the Hugging Face Hub

login(token=token)

print("Logged into huggingface")



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

tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

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



# Get the full path of the current file

current_file_path = os.path.abspath(__file__)

# Get the parent directory

parent_dir = os.path.dirname(current_file_path)

# Get the grandparent directory

grandparent_dir = os.path.dirname(parent_dir)

# Extract the name of the grandparent directory

grandparent_dir_name = os.path.basename(grandparent_dir)

# Split the grandparent directory name by underscores

parts = grandparent_dir_name.split('_')

# Take the last three components, if they exist

last_three_words = '_'.join(parts[-3:]) if len(parts) >= 3 else grandparent_dir_name

individual_file_path = '../Indi_Train_Eval'+str(last_three_words)+'.csv'



prev_round = 0

def check_base_model(prev_round):

    #check if the "round_"+prev_round+"/server/merged_model" directory exists

    directory_path = "round"+str(prev_round)+"/server/merged_model_Impr"

    # Check if the directory exists

    if os.path.exists(directory_path):

        print(f"The directory '{directory_path}' exists.")

    else:

        print(f"The directory '{directory_path}' does not exist.")
        #Download model from hugginface. 
        # model = LlamaForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache = False, device_map=device_map)
        # Load the model from Hugging Face
        #model = AutoModel.from_pretrained(model_id)
        # Save the model to the specified directory
        #from peft import AutoPeftModelForCausalLM
        adapter_dir = "round"+str(prev_round)+"/server/Ada_Impr"
        new_model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, #args.output_dir,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        ) #This is the adapters reloaded.
        merged_model = new_model.merge_and_unload()
        merged_model.save_pretrained(directory_path,safe_serialization=True)

        print(f"Model '{model_id}' has been saved to '{directory_path}'.")



def del_base_model(prev_round):

    #check if the "round_"+prev_round+"/server/merged_model" directory exists

    directory_path = "round"+str(prev_round)+"/server/merged_model_Impr"

    # Check if the directory exists

    if os.path.exists(directory_path):

        print(f"The directory '{directory_path}' exists.")

        shutil.rmtree(directory_path)

        print(f"The directory '{directory_path}' deleted.")



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

    return generated_docstring, sample['summary']



# generated, ground_truth = generate_docstring(sample, tokenizer)

def evaluate_docstring(merged_model, test_dataset, tokenizer):

    # Initialize evaluation metrics

    rouge = Rouge()

    all_generated = []

    all_ground_truth = []



    # Loop over the entire test dataset to generate docstrings

    for idx in range(len(test_dataset)):

        sample = test_dataset[idx]

        generated, ground_truth = generate_docstring(merged_model, sample, tokenizer)

        all_generated.append(generated)

        all_ground_truth.append(ground_truth)

    #Health check of generated data:

    for idx, text in enumerate(all_generated):

        if not text.strip():  # If text is empty or just whitespace

            print(f"Empty string found at index {idx}")

            all_generated[idx] = "NO_OUTPUT"

    for idx, text in enumerate(all_ground_truth):

        if not text.strip():  # If text is empty or just whitespace

            print(f"Empty reference found at index {idx}")

            all_ground_truth[idx] = "NO_INPUT"

            all_generated[idx] = "SO_NO_OUTPUT"



    # Evaluate Corpus-BLEU

    bleu_score = corpus_bleu([[gt.split()] for gt in all_ground_truth], [g.split() for g in all_generated])



    # Evaluate METEOR

    meteor_scores = [meteor_score([gt.split()], g.split()) for gt, g in zip(all_ground_truth, all_generated)]

    average_meteor = sum(meteor_scores) / len(meteor_scores)



    # Evaluate ROUGE

    rouge_scores = rouge.get_scores(all_generated, all_ground_truth, avg=True)



    print(f'Corpus-BLEU: {bleu_score}')

    print(f'Average METEOR: {average_meteor}')

    print(f'ROUGE: {rouge_scores}')



    # Return the results

    evaluation_results = {

        'BLEU': bleu_score,

        'METEOR': average_meteor,

        'ROUGE': rouge_scores

    }



    return evaluation_results



def eval_client(merged_model, tokenizer, this_client):

    if this_client!="server":

        test_dataset_dir = "../code_docstring_corpus_data_test_client"+str(this_client)+"/"

        #for every project in this directory

        for project_nm in os.listdir(test_dataset_dir):

            folder_path = os.path.join(test_dataset_dir, project_nm)

            if os.path.isdir(folder_path):

                print("Running for project", project_nm) 

                test_dataset = load_from_disk(test_dataset_dir+str(project_nm)+"/")

                metric_results = evaluate_docstring(merged_model, test_dataset, tokenizer)

                # Define the file path and column headers

                file_path = individual_file_path

                column_headers = ["Data", "Client","project", "BaseModelName", "EvalDataSize", "C-BLEU", "METEOR", "ROUGE-L"]

                # Check if the file already exists

                if not os.path.exists(file_path):

                    # Create the CSV file and write the headers

                    with open(file_path, 'w', newline='') as csvfile:

                        writer = csv.writer(csvfile)

                        writer.writerow(column_headers)

            #         print(f"File '{file_path}' created successfully.")

            #     else:

            #         print(f"File '{file_path}' already exists.")



                with open(file_path, 'a', newline='') as csvfile:

                    writer = csv.writer(csvfile)

                    this_row=["JavaOnly", this_client, project_nm, model_id, len(test_dataset), metric_results['BLEU'],  metric_results['METEOR'],metric_results['ROUGE']]

                    writer.writerow(this_row)

                    print("Updated for", this_client, project_nm)

        return "Updated"

    else: #server

        for this_client in range(0,num_clients):

            test_dataset_dir = "../code_docstring_corpus_data_test_client"+str(this_client)+"/"

            #for every project in this directory

            for project_nm in os.listdir(test_dataset_dir):

                folder_path = os.path.join(test_dataset_dir, project_nm)

                if os.path.isdir(folder_path):

                    print("Running for project", project_nm) 

                    test_dataset = load_from_disk(test_dataset_dir+str(project_nm)+"/")

                    metric_results = evaluate_docstring(merged_model, test_dataset, tokenizer)

                    # Define the file path and column headers

                    file_path = individual_file_path

                    column_headers = ["Data", "Client","project", "BaseModelName", "EvalDataSize", "C-BLEU", "METEOR", "ROUGE-L"]

                    # Check if the file already exists

                    if not os.path.exists(file_path):

                        # Create the CSV file and write the headers

                        with open(file_path, 'w', newline='') as csvfile:

                            writer = csv.writer(csvfile)

                            writer.writerow(column_headers)

                #         print(f"File '{file_path}' created successfully.")

                #     else:

                #         print(f"File '{file_path}' already exists.")



                    with open(file_path, 'a', newline='') as csvfile:

                        writer = csv.writer(csvfile)

                        this_row=["JavaOnly", "Fed@BEST ("+str(BEST_round)+")", project_nm, model_id, len(test_dataset), metric_results['BLEU'],  metric_results['METEOR'],metric_results['ROUGE']]

                        writer.writerow(this_row)

                    print("Updated for", this_client, project_nm)

        return "Updated"



def merge_lora2base(adapter_dir):
    print("Adapter dir:", adapter_dir)
    new_model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, low_cpu_mem_usage=True,

    return_dict=True, torch_dtype=torch.float16, device_map=device_map,) #This is the adapters reloaded.

    # Merge LoRA and base model

    merged_model = new_model.merge_and_unload()

    return merged_model



def this_r_W_experiment(adapter_dir, this_client):

    merged_model = merge_lora2base(adapter_dir)

    status = eval_client(merged_model, tokenizer, this_client)

    del merged_model

    if status!="Updated":

        print("ERROR: Updating validation history file failed for client", this_client)

    else:

        print("Validation history file updated for client", this_client )



    #remove directory

    # Check if the directory exists

    mm_dir = adapter_dir+"../merged_model_Impr/"

    if os.path.exists(mm_dir) and os.path.isdir(mm_dir):

        # Remove the directory and all its contents

        shutil.rmtree(mm_dir)

        print(f"The directory '{mm_dir}' has been removed.")

    else:

        print(f"The directory '{mm_dir}' does not exist or is not a directory.")

def check_and_create_baseDir(round_num):

    # Check if the directory exists
    print("BEST ROUND is declared as ", round_num)
    check_base_model(0) #For >=1 round_num
    for this_round in range(1, round_num): #For >=2 round_num
        adapter_dir = 'round'+str(this_round)+'/server'+'/merged_model_Impr/'
        if not os.path.exists(adapter_dir):
            #After generating for this_round, delete this_round-2 model
            print("Merged_model doesn't exist for round", this_round)
            ada_dir = 'round'+str(this_round)+'/server'+'/Ada_Impr/'
            this_merged_model = merge_lora2base(ada_dir)
            del_base_model(this_round-2)
            print("Prev but one model deleted", this_round-2)
            this_merged_model.save_pretrained(adapter_dir,safe_serialization=True) 
            print("Merged Model saved for round", this_round)
        #del_base_model(this_round-2)


#Evaluate all clients on test dataset    

check_base_model(prev_round)

print("Now, base model definitely exists")


print("\n\n----------------------------------------------------------------------")

print("Beginning Fed evaluations for server")

print("EXP Start time:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
BEST_round = 12
check_and_create_baseDir(BEST_round)
adapter_dir = 'round'+str(BEST_round)+'/server'+'/Ada_Impr/'

this_r_W_experiment(adapter_dir, "server")

print("SEeeeeee!! Validation experiment completed for server")

print("EXP End time:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

del_base_model(BEST_round-1)
if BEST_round>1: #if ==1 then only round0 needs to be deleted
    del_base_model(BEST_round-2)

print("Deleted the base merged_model")



import gc

gc.collect()

gc.collect()

torch.cuda.empty_cache() # PyTorch thing

gc.collect()










