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
from struq import jload

LR_CONFIG = {
    'llama-7b': {
        'lr':     '2e-5', 
        'dpo_lr': '2.0e-4', 
        'kto_lr': '8e-5', 
        'orpo_lr': '6.4e-4',
    },
    'llama-13b': {
        'lr':     '2e-5', 
        'dpo_lr': '2.0e-4', 
    },
    'Mistral-7B-v0.1': {
        'lr':     '2.5e-6',
        'dpo_lr': '1.4e-4', 
    },
    'Meta-Llama-3-8B': {
        'lr':     '2e-6', 
        'dpo_lr': '1.6e-4',
    }
}

def get_sft_cmd(model, attack, data_path, model_max_length):
    current_t = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()).strftime("%Y-%m-%d-%H-%M-%S")
    lr = LR_CONFIG[model.replace('-Instruct', '').split('/')[1].split('_')[0]]['lr']
    if '-Instruct' in model: 
        if 'Spcl' in attack or 'Text' in attack: attack = attack.replace(attack[:12], model.split('/')[-1])
    if 'llama-7b' in model.replace('-Instruct', ''):
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
    if 'llama-13b' in model.replace('-Instruct', ''):
        return f'python -m torch.distributed.run --nproc_per_node=4 --master_port={29550 + np.random.randint(0, 1000)} train.py \
            --model_name_or_path {model} \
            --data_path {data_path} \
            --bf16 True \
            --output_dir {model}_{attack}_{current_t} \
            --num_train_epochs 3 \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 16 \
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
    elif 'Mistral-7B-v0.1' in model.replace('-Instruct', ''):
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
    elif 'Llama-3' in model.replace('-Instruct', ''):
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
    lr = LR_CONFIG[base_model.replace('-Instruct', '').split('/')[1]][alignment + '_lr']
    if 'llama-7b' in base_model.replace('-Instruct', ''):
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
    if 'llama-13b' in base_model.replace('-Instruct', ''):
        return f'python -m torch.distributed.run  --nproc_per_node=4 --master_port={29550 + np.random.randint(0, 1000)} align.py \
            --model_name_or_path {model} \
            --data_path {data_path} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --logging_steps 1 \
            --per_device_train_batch_size 2 \
            --learning_rate {lr} \
            --tf32 True \
            --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
            --fsdp "full_shard auto_wrap" \
            --lr_scheduler_type "cosine" \
            --gradient_accumulation_steps 32 \
            --output_dir {model}_{alignment}_{attack}_{current_t} \
            --num_train_epochs 3 \
            --attack {attack} \
            --alignment {alignment}\
            --model_max_length {model_max_length}'
    elif 'Mistral-7B-v0.1' in base_model.replace('-Instruct', ''):
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
    elif 'Llama-3' in base_model.replace('-Instruct', ''):
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
    parser.add_argument('--align_attack', type=str, default=['NaiveCompletion'], nargs='+')
    parser.add_argument('--alignment', type=str, default='dpo', choices=['dpo', 'kto', 'orpo'])
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
        slurm_prefix = f"#!/bin/bash\n\n#SBATCH --nodes=1\n#SBATCH --time={args.time}:00:00\n#SBATCH --account differential -q differential_high\n#SBATCH --gpus-per-node=4\n#SBATCH --cpus-per-task=10\n#SBATCH --output={output_dir}/train_%j.out\n\nsource activate {args.env}\n"

        if args.print_cmd: print(slurm_prefix + cmd + '\n' * 5); exit()
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
            if not os.path.exists(output_dir): pass
            else:
                if not [x for x in os.listdir(output_dir) if '.out' not in x]: continue
            if args.do_sft or args.do_align: time.sleep(150)
            print(f"Scheduling tests for {output_dir}, {1+len(completed)}/{len(output_dirs)}.")

            num_gcg_sample_per_gpu = 10
            for attack in args.test_attack:
                log_dir = output_dir if os.path.exists(output_dir) else (output_dir + '-log')
                os.makedirs(log_dir, exist_ok=True)
                cmd = f'python test.py --model_name_or_path {output_dir} --attack {attack} --defense {args.defense} --data_path {args.test_data}'
                if attack == 'gcg': slurm_prefix = f"#!/bin/bash\n\n#SBATCH --nodes=1\n#SBATCH --time={int(np.round(0.7*num_gcg_sample_per_gpu))}:00:00\n#SBATCH --account differential -q differential_high\n#SBATCH --gpus-per-node=1\n#SBATCH --cpus-per-task=8\n#SBATCH --output={log_dir}/gcg_%j.out\n\nsource activate {args.env}\n"
                elif attack == 'advp': slurm_prefix = f"#!/bin/bash\n\n#SBATCH --nodes=1\n#SBATCH --time=72:00:00\n#SBATCH --account differential -q differential_high\n#SBATCH --gpus-per-node=2\n#SBATCH --cpus-per-task=10\n#SBATCH --output={log_dir}/advp_%j.out\n\nsource activate {args.env}\n"
                else: slurm_prefix = f"#!/bin/bash\n\n#SBATCH --nodes=1\n#SBATCH --time=0{args.time}:00:00\n#SBATCH --account differential -q differential_high\n#SBATCH --gpus-per-node=1\n#SBATCH --cpus-per-task=10\n#SBATCH --output={log_dir}/{attack}_%j.out\n\nsource activate {args.env}\n"
                
                if args.print_cmd: print(slurm_prefix + cmd + '\n' * 5)
                else: 
                    temporary_slurm_file = 'test_' + output_dir.replace('/', '_') + '.slurm'
                    if attack == 'gcg': 
                        num_sample = len([x for x in jload(args.test_data) if x['input'] != ''])
                        log_path = log_dir + '/gcg/len20_500step_bs512_seed0_l50_t1.0_static_k256'
                        if os.path.exists(log_path):
                            logs = sorted(os.listdir(log_path), key=lambda x: int(x.split('.')[0]))
                            to_run_sample_ids = [x for x in range(num_sample) if (str(x)+'.jsonl' not in logs)]
                            for i, gcg_log in enumerate(logs):
                                with open(os.path.join(log_path, gcg_log), 'r') as f:
                                    txt = f.read()
                                    if 'begin_with": true' not in txt and txt.count('{"loss":') != 26:
                                        to_run_sample_ids.append(int(gcg_log.split('.')[0]))
                            if to_run_sample_ids == []: print('GCG already completed for', log_dir); continue
                        else: to_run_sample_ids = list(range(num_sample))
                        
                        for i in range(len(to_run_sample_ids) // num_gcg_sample_per_gpu + 1):
                            sample_ids = ' '.join(map(str, to_run_sample_ids[i*num_gcg_sample_per_gpu:(i+1)*num_gcg_sample_per_gpu]))
                            if sample_ids == '': continue
                            with open(temporary_slurm_file, 'w') as f: f.write(slurm_prefix + cmd + ' --sample_ids ' + sample_ids)
                            os.system('sbatch ' + temporary_slurm_file)
                            os.remove(temporary_slurm_file)
                            print(temporary_slurm_file + '\n' * 3 + slurm_prefix + cmd + ' --sample_ids ' + sample_ids + '\n' * 10)
                            #time.sleep(1)
                    else:
                        with open(temporary_slurm_file, 'w') as f: f.write(slurm_prefix + cmd)
                        os.system('sbatch ' + temporary_slurm_file)
                        os.remove(temporary_slurm_file)
                        print(temporary_slurm_file + '\n' * 3 + slurm_prefix + cmd + '\n' * 10)
                        time.sleep(2)
            completed.append(output_dir)
        if args.print_cmd: print('Jobs not submitted!')
        else: time.sleep(test_scanning_interval)
        test_waiting_time += test_scanning_interval
        if test_waiting_time > 24 * 60 * 60: exit() # kill the python script after 24 hours

if __name__ == '__main__':
    train_and_test()