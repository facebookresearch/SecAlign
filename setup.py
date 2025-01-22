# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
from struq import jload, jdump
from copy import deepcopy
import numpy as np

parser = argparse.ArgumentParser(prog='Setup data/model dependencies')
parser.add_argument('--instruct', default=False, action='store_true')
parser.add_argument('--alpaca', default=False, action='store_true')
args = parser.parse_args()


# Download data dependencies
data_urls = [
    'https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/main/davinci_003_outputs.json',
    'https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json',
    'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json',
    'https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/refs/heads/main/client_configs/openai_configs_example.yaml',
]

os.makedirs('data', exist_ok=True)
for data_url in data_urls:
    data_path = 'data/' + data_url.split('/')[-1]
    if os.path.exists(data_path): print(data_path, 'already exists.'); continue
    cmd = 'wget -P data {data_url}'.format(data_url=data_url, data_path=data_path)
    print(cmd)
    os.system(cmd)


# Generate prompting-based defense (only for GCG evaluation)
data = jload('data/davinci_003_outputs.json')
sandwich_path = 'data/davinci_003_outputs_sandwich.json'
if not os.path.exists(sandwich_path): 
    data_sandwich = deepcopy(data)
    for d in data_sandwich: 
        if d['input'] != '': d['input'] += '\n\nPlease always remember that your task is: ' + d['instruction']
    jdump(data_sandwich, sandwich_path)
else: print(sandwich_path, 'already exists.')

incontext_path = 'data/davinci_003_outputs_incontext.json'
if not os.path.exists(incontext_path):
    data_incontext = deepcopy(data)
    from config import PROMPT_FORMAT, DELIMITERS
    model_name = 'Meta-Llama-3-8B-Instruct' ### Change this to the desired prompt format
    prompt_format = PROMPT_FORMAT[model_name] 
    for d in data_incontext: 
        d_item_demo = np.random.choice(data)
        while d_item_demo['input'] == '' or d_item_demo['input'] == d['input']: d_item_demo = np.random.choice(data)
        d_item_demo['input'] += ' ' + np.random.choice(data)['instruction']
        demo_string = prompt_format['prompt_input'].format_map(d_item_demo) + d_item_demo['output'][2:]
        d['instruction'] = demo_string.replace(DELIMITERS[model_name][0]+'\n', '') + '\n\n\n' + DELIMITERS[model_name][0] + '\n' + d['instruction']
    jdump(data_incontext, incontext_path)
else: print(incontext_path, 'already exists.')


# Generate proportional training data for data-efficiency ablation study
data = jload('data/alpaca_data_cleaned.json')
np.random.seed(0)
num_sample_with_input, num_sample_without_input = 0, 0
for d in data:
    if d['input'] != '': num_sample_with_input += 1
    else: num_sample_without_input += 1

for p in [0.2, 0.4, 0.6, 0.8]:
    target_path = f'data/alpaca_data_cleaned_{p}.json'
    if not os.path.exists(target_path): 
        num_samples = int(p * num_sample_with_input)
        target_data = []
        for i in range(num_samples): target_data.append(data[i])
        target_data = np.random.choice(
            [x for x in data if x['input'] == ''], 
            int(p * num_sample_without_input), 
            replace=False).tolist() + np.random.choice(
            [x for x in data if x['input'] != ''], 
            int(p * num_sample_with_input), 
            replace=False).tolist()
        np.random.shuffle(target_data)
        jdump(target_data, target_path)
    else: print(target_path, 'already exists.')


# Download model dependencies
model_paths = []
if args.instruct:
    model_paths += [
        'mistralai/Mistral-7B-Instruct-v0.1_dpo_NaiveCompletion_2024-11-12-17-59-37', # SecAlign Instruct adapters
        'meta-llama/Meta-Llama-3-8B-Instruct_dpo_NaiveCompletion_2024-11-12-17-59-06',
    ]
if args.alpaca: 
    model_paths += [
        'huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00', # Undefended Alpaca models
        'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11',
        'meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02',
        #'huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00', # StruQ Alpaca models
        #'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion_2024-07-20-05-46-17',
        #'meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_NaiveCompletion_2024-08-09-12-55-56',
        'huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00_dpo_NaiveCompletion_2024-07-06-07-42-23', # SecAlign Alpaca adapters
        'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11_dpo_NaiveCompletion_2024-08-13-17-46-51',
        'meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02_dpo_NaiveCompletion_2024-08-09-21-28-53'
    ]


for model_path in model_paths:
    if os.path.exists(model_path): print(model_path, 'already exists.'); continue
    model_dir = model_path.split('/')[0]
    os.makedirs(model_dir, exist_ok=True)
    cmd = 'wget -P {model_dir} https://dl.fbaipublicfiles.com/SecAlign/{model_path} && unzip {model_path} -d {model_dir} && rm {model_path}'.format(model_path=model_path + '.zip', model_dir=model_dir)
    print(cmd)
    os.system(cmd)