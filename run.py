# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import argparse
import os
import time
import numpy as np
import datetime
import re

LR_CONFIG = {
    'llama-7b': {
        'lr':     '2e-5', ###
        'dpo_lr': '2.0e-4', # '2.8e-4' # 2.2 2.4 2.6 2.8 3.0
        'kto_lr': '8e-5',   # 0.8 1.4 1.6 1.8 2.5
        'orpo_lr': '6.4e-4',# 64, 32, 16, 8, 4
    },
    'Mistral-7B-v0.1': {
        'lr':     '2.5e-6',
        'dpo_lr': '1.4e-4', #0.8 1.4 1.6 1.8 2.5 #  # 2, 1.2, 0.8, 0.4, 0.1
    },
    'Meta-Llama-3-8B': {
        'lr':     '2e-6', # 1 2 4 8 16 # 0.6 0.4 0.2 0.1 0.05 0.02   # 10 8 6 4 2 1 0.8 0.6 0.4
        'dpo_lr': '1.6e-4',
    },
    'vicuna-7b-v1.5': {
        'lr':     '2e-5',
        'dpo_lr': '1.6e-4',
    },
}

def get_sft_cmd(model, attack, data_path, model_max_length):
    current_t = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()).strftime("%Y-%m-%d-%H-%M-%S")
    lr = LR_CONFIG[model.split('/')[1]]['lr']
    if 'llama-7b' in model or 'vicuna-7b-v1.5' in model:
        return f'python -m torch.distributed.run --nproc_per_node=4 --master_port={29550 + np.random.randint(0, 1000)} train.py \
            --model_name_or_path {model} \
            --data_path {data_path} \
            --bf16 True \
            --output_dir {model}_{attack}_{current_t} \
            --num_train_epochs 3 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --save_total_limit 1 \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --fsdp "full_shard auto_wrap" \
            --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
            --tf32 True\
            --attack {attack}\
            --model_max_length {model_max_length}'
    elif 'Mistral-7B-v0.1' in model:
        return f'python -m torch.distributed.run --nproc_per_node=4 --master_port={29550 + np.random.randint(0, 1000)} train.py \
            --model_name_or_path {model} \
            --window_size 256 \
            --padding_side left \
            --data_path {data_path} \
            --bf16 True \
            --output_dir {model}_{attack}_{current_t} \
            --num_train_epochs 3 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --save_total_limit 1 \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --fsdp "full_shard auto_wrap" \
            --fsdp_transformer_layer_cls_to_wrap "MistralDecoderLayer" \
            --tf32 True\
            --attack {attack}\
            --lr_scale True \
            --downsample True\
            --model_max_length {model_max_length}'
    elif 'Meta-Llama-3' in model:
        return f'python -m torch.distributed.run --nproc_per_node=4 --master_port={29550 + np.random.randint(0, 1000)} train.py \
            --model_name_or_path {model} \
            --data_path {data_path} \
            --bf16 True \
            --output_dir {model}_{attack}_{current_t} \
            --num_train_epochs 3 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --save_total_limit 1 \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --fsdp "full_shard auto_wrap" \
            --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
            --tf32 True\
            --attack {attack}\
            --model_max_length {model_max_length}'
    else: raise NotImplementedError

def get_align_cmd(model, attack, alignment, data_path, model_max_length):
    base_model = model.split('/')[-2] + '/' + model.split('/')[-1].split('_')[0]
    current_t = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()).strftime("%Y-%m-%d-%H-%M-%S")
    lr = LR_CONFIG[base_model.split('/')[1]][alignment + '_lr']
    if 'llama-7b' in base_model or 'vicuna-7b-v1.5' in base_model:
        return f'python -m torch.distributed.run  --nproc_per_node=4 --master_port={29550 + np.random.randint(0, 1000)} align.py \
            --model_name_or_path {model} \
            --data_path {data_path} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --logging_steps 1 \
            --per_device_train_batch_size 4 \
            --learning_rate {lr} \
            --tf32 True \
            --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
            --fsdp "full_shard auto_wrap" \
            --lr_scheduler_type "cosine" \
            --gradient_accumulation_steps 16 \
            --output_dir {model}_{alignment}_{attack}_{current_t} \
            --num_train_epochs 3 \
            --attack {attack} \
            --alignment {alignment}\
            --model_max_length {model_max_length}'
    elif 'Mistral-7B-v0.1' in base_model:
        return f'python -m torch.distributed.run --nproc_per_node=4 --master_port={29550 + np.random.randint(0, 1000)} align.py \
            --model_name_or_path {model} \
            --window_size 256 \
            --padding_side left \
            --data_path {data_path} \
            --output_dir {model}_{alignment}_{attack}_{current_t} \
            --num_train_epochs 3 \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 16 \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --save_total_limit 1 \
            --learning_rate {lr} \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --fsdp "full_shard auto_wrap" \
            --fsdp_transformer_layer_cls_to_wrap "MistralDecoderLayer" \
            --tf32 True \
            --attack {attack} \
            --lr_scale True \
            --downsample True \
            --alignment {alignment}\
            --model_max_length {model_max_length}'
    elif 'Meta-Llama-3' in base_model:
        return f'python -m torch.distributed.run  --nproc_per_node=4 --master_port={29550 + np.random.randint(0, 1000)} align.py \
            --model_name_or_path {model} \
            --data_path {data_path} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --logging_steps 1 \
            --per_device_train_batch_size 4 \
            --learning_rate {lr} \
            --tf32 True \
            --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
            --fsdp "full_shard auto_wrap" \
            --lr_scheduler_type "cosine" \
            --gradient_accumulation_steps 16 \
            --output_dir {model}_{alignment}_{attack}_{current_t} \
            --num_train_epochs 3 \
            --attack {attack} \
            --alignment {alignment}\
            --model_max_length {model_max_length}'
    else: raise NotImplementedError

def train_and_test():
    parser = argparse.ArgumentParser(prog='Training model(s) accepting structured queries on 4 80GB A100s', description='The script implements the slurm pipeline for training multiple defended models and later testing them with multiple attacks.')
    parser.add_argument('-m', '--model', type=str, nargs="+")
    parser.add_argument('--data', type=str, default='data/alpaca_data_cleaned.json')
    parser.add_argument('--test_data', type=str, default='data/davinci_003_outputs.json')
    parser.add_argument('--sft_attack', type=str, default=['SpclSpclSpcl_NaiveCompletion'], nargs='+')
    parser.add_argument('--align_attack', type=str, default=['Naive'], nargs='+')
    parser.add_argument('--alignment', type=str, default='dpo')
    parser.add_argument('--test_attack', type=str, default=['none', 'ignore', 'completion_real', 'completion_realcmb', 'gcg', 'advp'], nargs='+') 
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('-d', '--defense', type=str, default='none', choices=['none', 'sandwich', 'instructional', 'reminder', 'isolation', 'incontext'], help='Baseline test-time zero-shot prompting defense')
    parser.add_argument('-t', '--time', type=float, default=6)
    parser.add_argument('-e', '--env', type=str, default='secalign')
    parser.add_argument('--do_sft', default=False, action='store_true')
    parser.add_argument('--do_align', default=False, action='store_true')
    parser.add_argument('--do_test', default=False, action='store_true')
    parser.add_argument('--print_cmd', default=False, action='store_true')
    args = parser.parse_args()
    
    if args.do_sft and args.do_align: raise NotImplementedError
    if args.do_sft and not args.do_align: 
        attack_list = args.sft_attack
        output_dirs = []
    if not args.do_sft and args.do_align:
        attack_list = args.align_attack
        output_dirs = []
    if not args.do_sft and not args.do_align:
        attack_list = []
        output_dirs = args.model
    if len(args.model) == 1: args.model = args.model[0]

    for attack in attack_list:
        if args.do_sft:   cmd = get_sft_cmd(args.model, attack, args.data, args.model_max_length)  
        if args.do_align: cmd = get_align_cmd(args.model, attack, args.alignment, args.data, args.model_max_length)
        output_dir = re.search(f'--output_dir (.+?)--num_train_epochs', cmd).group(1).replace(' ', '')
        os.makedirs(output_dir, exist_ok=True)
        if args.alignment == 'kto': args.time *= 4
        slurm_prefix = f"#!/bin/bash\n\n#SBATCH --nodes=1\n#SBATCH --time={args.time}:00:00\n#SBATCH --partition=learnlab\n#SBATCH --gpus-per-node=4\n#SBATCH --cpus-per-task=48\n#SBATCH -C ampere80gb\n#SBATCH --output={output_dir}/train_%j.out\n\nsource activate {args.env}\n"

        if args.print_cmd: print('\n' + cmd + '\n' * 10); exit()
        else: 
            temporary_slurm_file = f'train_' + output_dir.replace('/', '_') + '.slurm'
            with open(temporary_slurm_file, 'w') as f: f.write(slurm_prefix + cmd)
            os.system('sbatch ' + temporary_slurm_file)
            os.remove(temporary_slurm_file)
            print(temporary_slurm_file + '\n' * 3 + slurm_prefix + cmd + '\n' * 10)
        output_dirs.append(output_dir)
        time.sleep(2)
    
    if not args.do_test: return
    if args.do_sft or args.do_align: print("Submitted all", len(output_dirs), "job(s), waiting for completion...")
    completed = []

    test_waiting_time = 0
    test_scanning_interval = 10
    while len(completed) < len(output_dirs):
        for output_dir in [x for x in output_dirs if x not in completed]:
            if not [x for x in os.listdir(output_dir) if '.out' not in x]: continue
            if args.do_sft or args.do_align: time.sleep(150)
            print(f"Scheduling tests for {output_dir}, {1+len(completed)}/{len(output_dirs)}.")

            for attack in args.test_attack:
                if attack in ['advp', 'gcg']:
                    cmd = 'python run_advp_gcg.py -c test -a ' + attack + ' -m ' + output_dir + ' --data_path ' + args.test_data
                    print(cmd + '\n' * 10)
                    if args.print_cmd: print('\n' + cmd + '\n' * 10); exit()
                    else: os.system(cmd)
                else:
                    gpu = 'ampere80gb' if attack == 'none' else 'volta32gb'
                    slurm_prefix = f"#!/bin/bash\n\n#SBATCH --nodes=1\n#SBATCH --time=0{args.time}:00:00\n#SBATCH --partition=learnlab\n#SBATCH --gpus-per-node=1\n#SBATCH --cpus-per-task=48\n#SBATCH -C {gpu}\n#SBATCH --output={output_dir}/{attack}_%j.out\n\nsource activate {args.env}\n"
                    cmd = f'python test.py --model_name_or_path {output_dir} --attack {attack} --defense {args.defense} --data_path {args.test_data}'
                    
                    if args.print_cmd: print('\n' + cmd + '\n' * 10); exit()
                    else: 
                        temporary_slurm_file = 'test_' + output_dir.replace('/', '_') + '.slurm'
                        with open(temporary_slurm_file, 'w') as f: f.write(slurm_prefix + cmd)
                        os.system('sbatch ' + temporary_slurm_file)
                        os.remove(temporary_slurm_file)
                        print(temporary_slurm_file + '\n' * 3 + slurm_prefix + cmd + '\n' * 10)
                time.sleep(2)
            completed.append(output_dir)
        time.sleep(test_scanning_interval)
        test_waiting_time += test_scanning_interval
        if test_waiting_time > 24 * 60 * 60: exit() # kill the python script after 24 hours

if __name__ == '__main__':
    train_and_test()