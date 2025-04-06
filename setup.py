# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
from struq import jload, jdump
from copy import deepcopy
import numpy as np
from config import PROMPT_FORMAT, DELIMITERS

parser = argparse.ArgumentParser(prog='Setup data/model dependencies')
parser.add_argument('--instruct', default=False, action='store_true')
parser.add_argument('--alpaca', default=False, action='store_true')
args = parser.parse_args()


# Download data dependencies
data_urls = [
    'https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/main/davinci_003_outputs.json',
    'https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json',
    'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json',
    'https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/refs/heads/main/client_configs/openai_configs_example.yaml'
]

os.makedirs('data', exist_ok=True)
for data_url in data_urls:
    data_path = 'data/' + data_url.split('/')[-1]
    if os.path.exists(data_path): print(data_path, 'already exists.'); continue
    cmd = 'wget -P data {data_url}'.format(data_url=data_url, data_path=data_path)
    print(cmd)
    os.system(cmd)


# Process in-context demonstration inputs
data = jload('data/davinci_003_outputs.json')
for model_name in ['Meta-Llama-3-8B-Instruct', 'Mistral-7B-Instruct-v0.1']:
    incontext_path = 'data/davinci_003_outputs_incontext_%s.json' % model_name
    if not os.path.exists(incontext_path):
        data_incontext = deepcopy(data)
        prompt_format = PROMPT_FORMAT[model_name] 
        for d in data_incontext: 
            if d['input'] == '': continue
            d_item_demo = np.random.choice(data)
            while d_item_demo['input'] == '' or d_item_demo['input'] == d['input']: d_item_demo = np.random.choice(data)
            d_item_demo['input'] += ' ' + np.random.choice(data)['instruction']
            demo_string = prompt_format['prompt_input'].format_map(d_item_demo) + d_item_demo['output'][2:]
            d['instruction'] = demo_string.replace(DELIMITERS[model_name][0]+'\n', '') + '\n\n' + DELIMITERS[model_name][0].replace('<|begin_of_text|>', '') + '\n' + d['instruction']
        jdump(data_incontext, incontext_path)
    else: print(incontext_path, 'already exists.')


# Download model dependencies
model_paths = []
if args.instruct:
    model_paths += [
        'mistralai/Mistral-7B-Instruct-v0.1_dpo_NaiveCompletion_2025-03-12-12-01-27', # SecAlign Instruct adapters
        'meta-llama/Meta-Llama-3-8B-Instruct_dpo_NaiveCompletion_2024-11-12-17-59-06-resized',
        #'mistralai/Mistral-7B-Instruct-v0.1_Mistral-7B-Instruct-v0.1_NaiveCompletion_2025-03-12-12-01-27', # StruQ Instruct models
        #'meta-llama/Meta-Llama-3-8B-Instruct_Meta-Llama-3-8B-Instruct_NaiveCompletion_2025-03-18-06-14-30-lr6e-6'
        
        ##'mistralai/Mistral-7B-Instruct-v0.1_dpo_NaiveCompletion_2024-11-12-17-59-37', # SecAlign Instruct adapters
        ##'meta-llama/Meta-Llama-3-8B-Instruct_dpo_NaiveCompletion_2024-11-12-17-59-06',
        ##'mistralai/Mistral-7B-Instruct-v0.1_Mistral-7B-Instruct-v0.1_NaiveCompletion_2024-11-12-17-59-27', # StruQ Instruct models
        ##'meta-llama/Meta-Llama-3-8B-Instruct_Meta-Llama-3-8B-Instruct_NaiveCompletion_2024-11-12-17-58-38'
    ]
if args.alpaca: 
    model_paths += [
        'huggyllama/llama-7b_SpclSpclSpcl_None_2025-03-12-01-01-20', # Undefended Alpaca models
        'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08',
        'meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2025-03-12-01-02-14',
        #'huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2025-03-12-01-02-37', # StruQ Alpaca models
        #'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion_2025-03-15-03-25-16',
        #'meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_NaiveCompletion_2025-03-18-06-16-46-lr4e-6',
        'huggyllama/llama-7b_SpclSpclSpcl_None_2025-03-12-01-01-20_dpo_NaiveCompletion_2025-03-12-05-33-03', # SecAlign Alpaca adapters
        'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08_dpo_NaiveCompletion_2025-03-14-18-26-14',
        'meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2025-03-12-01-02-14_dpo_NaiveCompletion_2025-03-12-05-33-03'

        ##'huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00', # Undefended Alpaca models
        ##'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11',
        ##'meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02',
        ###'huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00', # StruQ Alpaca models
        ###'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion_2024-07-20-05-46-17',
        ###'meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_NaiveCompletion_2024-08-09-12-55-56',
        ##'huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00_dpo_NaiveCompletion_2024-07-06-07-42-23', # SecAlign Alpaca adapters
        ##'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11_dpo_NaiveCompletion_2024-08-13-17-46-51',
        ##'meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02_dpo_NaiveCompletion_2024-08-09-21-28-53'
    ]
 

for model_path in model_paths:
    if os.path.exists(model_path): print(model_path, 'already exists.'); continue
    model_dir = model_path.split('/')[0]
    os.makedirs(model_dir, exist_ok=True)
    cmd = 'wget -P {model_dir} https://dl.fbaipublicfiles.com/SecAlign/{model_path} && unzip {model_path} -d {model_dir} && rm {model_path}'.format(model_path=model_path + '.zip', model_dir=model_dir)
    print(cmd)
    os.system(cmd)