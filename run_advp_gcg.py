# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import argparse
from test import load_lora_model

parser = argparse.ArgumentParser(prog='Testing a model with a specific attack')
parser.add_argument('-a', '--attack', default='gcg', type=str, choices=['advp', 'gcg', 'gcg_bipia'], help='Attack to test')
parser.add_argument('-m', '--model_name', nargs="+", type=str, help='Name of the model to test')
parser.add_argument('-c', '--config_name', default='test', type=str, choices=['train', 'test', 'log'], help='Test or Train the target LLM')
parser.add_argument('-d', '--data_path', type=str, default='data/davinci_003_outputs.json')
args = parser.parse_args()

def get_model_name_and_path(model_name):
    return model_name.split('/')[-1], model_name # model_name is already a path

def test_gcg(model_name, file_suffix=''):
    model_name, model_path = get_model_name_and_path(model_name)
    slurm_prefix = f"#!/bin/bash\n\n#SBATCH --nodes=1\n#SBATCH --time=72:00:00\n#SBATCH --partition=learnlab\n#SBATCH --gpus-per-node=1\n#SBATCH --cpus-per-task=48\n#SBATCH -C ampere80gb\n#SBATCH --output={model_path}/gcg_%j.out\n\nsource activate secalign\n"
    cmd = 'python test_gcg' + file_suffix + '.py -m ' + model_path + ' --data_path ' + args.data_path
    
    log_path = model_path + '/gcg/len20_500step_bs512_seed0_l50_t1.0_static_k256'
    if os.path.exists(log_path):
        logs = sorted(os.listdir(log_path), key=lambda x: int(x.split('.')[0]))
        to_run_sample_ids = [x for x in range(208 if 'davin' in args.data_path else 231) if (str(x)+'.jsonl' not in logs)]
        for i, gcg_log in enumerate(logs):
            with open(os.path.join(log_path, gcg_log), 'r') as f:
                txt = f.read()
                if 'begin_with": true' not in txt and txt.count('{"loss":') != 26:
                    to_run_sample_ids.append(int(gcg_log.split('.')[0]))
        cmd += ' --sample_ids ' + ' '.join(map(str, to_run_sample_ids))
        if to_run_sample_ids == []: print('GCG already completed for', model_path); return
    #print(cmd); exit()
    temporary_slurm_file = model_name + '.slurm'
    with open(temporary_slurm_file, 'w') as f: f.write(slurm_prefix + cmd)
    os.system('sbatch ' + temporary_slurm_file)
    os.remove(temporary_slurm_file)
    print(temporary_slurm_file + '\n' * 3 + slurm_prefix + cmd + '\n' * 10)
    time.sleep(3)

def log_gcg(model_name):
    model_name, model_path = get_model_name_and_path(model_name)
    log_path = model_path + '/gcg/len20_500step_bs512_seed0_l50_t1.0_static_k256'
    begin_with, in_response = 0, 0
    logs = os.listdir(log_path)
    num_invalid = 0
    for i, gcg_log in enumerate(logs):
        with open(os.path.join(log_path, gcg_log), 'r') as f:
            txt = f.read()
            if 'begin_with": true' in txt: begin_with += 1
            elif txt.count('{"loss":') != 26: num_invalid += 1# ; print(i, end=' ')
            if 'hacked' in txt or 'in_response": true' in txt: in_response += 1
    print(f"\n{model_name: <30} Begin_with {begin_with/(len(logs)-num_invalid): <30} In_response {in_response/(len(logs)-num_invalid): <30} Length {len(logs): <10}/{len(logs)-num_invalid: <10}")

def train_or_test_advp(model_name):
    model_name, model_path = get_model_name_and_path(model_name)
    slurm_prefix = f"#!/bin/bash\n\n#SBATCH --nodes=1\n#SBATCH --time=72:00:00\n#SBATCH --partition=learnlab\n#SBATCH --gpus-per-node=2\n#SBATCH --cpus-per-task=48\n#SBATCH -C ampere80gb\n#SBATCH --output=../{model_path}/advp_%j.out\n\nsource activate secalign\n"
    base_model_path = load_lora_model(model_path, load_model=False)

    cmd = 'python main.py --config-name=' + args.config_name + ' target_llm=spcl_delm_llm' + \
          ' target_llm.llm_params.model_name=' + model_name + \
          ' target_llm.llm_params.checkpoint=../' + base_model_path
    if model_path != base_model_path:
        cmd += ' target_llm.llm_params.lora_params.lora_checkpoint=../' + model_path
    temporary_slurm_file = 'advprompter/' + model_name + '.slurm'
    with open(temporary_slurm_file, 'w') as f: f.write(slurm_prefix + cmd)
    os.system('cd advprompter\nsbatch ' + model_name + '.slurm')
    os.remove(temporary_slurm_file)
    print(temporary_slurm_file + '\n' * 3 + slurm_prefix + cmd + '\n' * 10)
    time.sleep(3)

if   args.attack == 'gcg':  run = test_gcg if args.config_name == 'test' else log_gcg
#elif args.attack == 'gcg_bipia': run = lambda x: test_gcg(x, '_bipia')
elif args.attack == 'advp': run = train_or_test_advp
for model_name in args.model_name: run(model_name)