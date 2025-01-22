# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import cv2, os
import numpy as np
import argparse
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams['font.size'] = 14

save_dir = 'figures'
os.makedirs(save_dir, exist_ok=True)

colors = sns.color_palette("deep")
colors = {'None': colors[7], 'StruQ': colors[9], 'SecAlign': colors[1],
          'No defense': colors[7], 'SOTA fine-tuning-based defense': colors[9], 'SOTA prompting-based defense': colors[8], 'SecAlign fine-tuning-based defense': colors[1]}


def motivation():
    def extract_log_dict(log_file): 
        with open(log_file, 'r', encoding='utf-8') as f: return [json.loads(x[:-1].replace('\'', '\"')) for x in f.readlines() if '{\'loss\':' in x]
    def avg_firsts(ls, n): return sum(ls[:n]) / n 

    ref_log = extract_log_dict('data/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00_dpo_NaiveCompletion_2024-07-09-07-55-19_train_29923141.out')
    ref_logps_chosen = [x['logps/chosen'] for x in ref_log]
    ref_logps_rejected = [x['logps/rejected'] for x in ref_log]
    target_log = extract_log_dict('data/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00_dpo_NaiveCompletion_2024-08-15-02-26-16_train_31165292.out')
    target_logps_chosen = [x['logps/chosen'] for x in target_log]
    target_logps_rejected = [x['logps/rejected'] for x in target_log]
    n = 3
    plt.figure(figsize=(6, 7))#plt.figure(figsize=(8, 5))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('SecAlign training step(s)', fontsize=18)
    plt.ylabel('Log probability of output', fontsize=18)
    plt.plot(target_logps_chosen, linewidth=2, color=colors['SecAlign'], linestyle='--', label='SecAlign (desirable output)')
    plt.plot(target_logps_rejected, linewidth=2, color=colors['SecAlign'], label='SecAlign (undesirable output)')
    plt.plot([avg_firsts(ref_logps_chosen, n)] * len(target_logps_chosen), linewidth=2, color=colors['StruQ'], linestyle='--', label='StruQ (desirable output)')
    plt.plot([avg_firsts(ref_logps_rejected, n)] * len(target_logps_chosen), linewidth=2, color=colors['StruQ'], label='StruQ (undesirable output)')
    plt.legend(framealpha=0.5, loc='center right')
    plt.savefig(save_dir + '/motivation.pdf', dpi=1000, bbox_inches='tight')
    plt.clf()


def main():
    # python log.py -m huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00 huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00 huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00_dpo_NaiveCompletion_2024-07-06-07-42-23 mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11 mistralai/Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion_2024-07-20-05-46-17 mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11_dpo_NaiveCompletion_2024-08-13-17-46-51 meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02 meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_NaiveCompletion_2024-08-09-12-55-56 meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02_dpo_NaiveCompletion_2024-08-09-21-28-53
    main_results = {
        'Llama-7B': { # 'llama-7b'
            'None': {
                'AlpacaEval2\nWinRate (↑)': 55.46,
                'Max ASR (↓)\nOpt.-Free': 75,
                #'AdvPrompter\nASR (↓)': 60,
                'Max ASR (↓)\nOpt.-Based': 97,
                },
            'StruQ': {
                'AlpacaEval2\nWinRate (↑)': 54.55,
                'Max ASR (↓)\nOptimization-Free': 0.5,
                #'AdvPrompter\nASR (↓)': 4,
                'Max ASR (↓)\nOptimization-Based': 60, #58,
                },
            'SecAlign': {
                'AlpacaEval2\nWinRate (↑)': 56.06,
                'Max ASR (↓)\nOptimization-Free': 0,
                #'AdvPrompter\nASR (↓)': 1,
                'Max ASR (↓)\nOptimization-Based': 14, #15,
                },
            },
        'Mistral-7B': { # 'Mistral-7B-v0.1'
            'None': {
                'AlpacaEval2\nWinRate (↑)': 72.21,
                'Max ASR (↓)\nOpt.-Free': 89,
                #'AdvPrompter\nASR (↓)': 72,
                'Max ASR (↓)\nOpt.-Based': 95,#99,
                },
            'StruQ': {
                'AlpacaEval2\nWinRate (↑)': 72.17,
                'Max ASR (↓)\nOptimization-Free': 4,
                #'AdvPrompter\nASR (↓)': 7,
                'Max ASR (↓)\nOptimization-Based': 41, #56,
                },
            'SecAlign': {
                'AlpacaEval2\nWinRate (↑)': 72.88,
                'Max ASR (↓)\nOptimization-Free': 0,
                #'AdvPrompter\nASR (↓)': 0,
                'Max ASR (↓)\nOptimization-Based': 1, #2,
                },
            },
        'Llama3-8B': { # Meta-Llama-3-8B
            'None': {
                'AlpacaEval2\nWinRate (↑)': 69.47,
                'Max ASR (↓)\nOpt.-Free': 90,
                #'AdvPrompter\nASR (↓)': 95,
                'Max ASR (↓)\nOpt.-Based': 98, #95, #89,
                },
            'StruQ': {
                'AlpacaEval2\nWinRate (↑)': 68.77,
                'Max ASR (↓)\nOptimization-Free': 0,
                #'AdvPrompter\nASR (↓)': 18,
                'Max ASR (↓)\nOptimization-Based': 43, #33,
                },
            'SecAlign': {
                'AlpacaEval2\nWinRate (↑)': 68.87,
                'Max ASR (↓)\nOptimization-Free': 0,
                #'AdvPrompter\nASR (↓)': 0,
                'Max ASR (↓)\nOptimization-Based': 9, #11,
                },
            }
        }

    width = 0.4
    #r1 = np.arange(len(main_results['Llama-7B']['None']))
    r1 = np.array([0, 1.2, 2.2])
    barwidth = len(r1) * 0.4 + width
    r1 = r1 * barwidth
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r = [r1, r2, r3]
    
    plt.figure(figsize=(22,6))
    for i, model_name in enumerate(main_results.keys()):
        plt.subplot(1, 3, i+1)
        for j, defense in enumerate(main_results[model_name]):
            plt.bar(r[j], main_results[model_name][defense].values(), color=colors[defense], label=defense, width=width)
            for k, v in enumerate(main_results[model_name][defense].values()): 
                if v == 0: plt.text(r[j][k], v + 1, str(v) + '%', ha='center')
                elif v < 1: plt.text(r[j][k], v + 1, '.5%', ha='center')
                elif v == 1: plt.text(r[j][k], v + 1, '1%', ha='center')
        plt.gca().axvline(x=(r1[0] + r1[1])/2 + width, color='grey', linewidth=1)
        plt.xticks([k + width for k in r1], main_results[list(main_results.keys())[0]]['None'].keys(), fontsize=16)
        plt.yticks(fontsize=18)
        plt.ylim(0, 100)
        if not i:
            plt.ylabel('WinRate / ASR (%)', fontsize=20)
            plt.legend(loc='upper left', framealpha=0.5, fontsize=18)
        plt.title(model_name, fontsize=24)
    plt.savefig(save_dir + f'/main_figure.pdf', dpi=1000, bbox_inches='tight')
    plt.clf()

def main_more():
    # python log.py -m huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00 huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00 huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00_dpo_NaiveCompletion_2024-07-06-07-42-23 mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11 mistralai/Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion_2024-07-20-05-46-17 mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11_dpo_NaiveCompletion_2024-08-13-17-46-51 meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02 meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_NaiveCompletion_2024-08-09-12-55-56 meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02_dpo_NaiveCompletion_2024-08-09-21-28-53
    main_results = {
        'Yi-1.5-6B': { # 'Mistral-7B-v0.1'
            'None': {
                'AlpacaEval2\nWinRate (↑)': 55.63,
                'Max ASR (↓)\nOpt.-Free': 84,
                #'AdvPrompter\nASR (↓)': 72,
                'Max ASR (↓)\nOpt.-Based': 97,#99,
                },
            'StruQ': {
                'AlpacaEval2\nWinRate (↑)': 56.44,
                'Max ASR (↓)\nOptimization-Free': 0,
                #'AdvPrompter\nASR (↓)': 7,
                'Max ASR (↓)\nOptimization-Based': 53, #56,
                },
            'SecAlign': {
                'AlpacaEval2\nWinRate (↑)': 54.64,
                'Max ASR (↓)\nOptimization-Free': 0,
                #'AdvPrompter\nASR (↓)': 0,
                'Max ASR (↓)\nOptimization-Based': 18, #2,
                },
            },
        'Llama-13B': { # Meta-Llama-3-8B
            'None': {
                'AlpacaEval2\nWinRate (↑)': 66.65,
                'Max ASR (↓)\nOpt.-Free': 75,
                #'AdvPrompter\nASR (↓)': 95,
                'Max ASR (↓)\nOpt.-Based': 98, #95, #89,
                },
            'StruQ': {
                'AlpacaEval2\nWinRate (↑)': 63.81,
                'Max ASR (↓)\nOptimization-Free': 19,
                #'AdvPrompter\nASR (↓)': 18,
                'Max ASR (↓)\nOptimization-Based': 25, #33,
                },
            'SecAlign': {
                'AlpacaEval2\nWinRate (↑)': 64.68,
                'Max ASR (↓)\nOptimization-Free': 0,
                #'AdvPrompter\nASR (↓)': 0,
                'Max ASR (↓)\nOptimization-Based': 3, #11,
                },
            }
        }

    width = 0.4
    #r1 = np.arange(len(main_results['Llama-7B']['None']))
    r1 = np.array([0, 1.2, 2.2])
    barwidth = len(r1) * 0.4 + width
    r1 = r1 * barwidth
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r = [r1, r2, r3]
    
    plt.figure(figsize=(16,6))
    for i, model_name in enumerate(main_results.keys()):
        plt.subplot(1, 2, i+1)
        for j, defense in enumerate(main_results[model_name]):
            plt.bar(r[j], main_results[model_name][defense].values(), color=colors[defense], label=defense, width=width)
            for k, v in enumerate(main_results[model_name][defense].values()): 
                if v == 0: plt.text(r[j][k], v + 1, str(v) + '%', ha='center')
                elif v < 1: plt.text(r[j][k], v + 1, '.5%', ha='center')
        plt.gca().axvline(x=(r1[0] + r1[1])/2 + width, color='grey', linewidth=1)
        plt.xticks([k + width for k in r1], main_results[list(main_results.keys())[0]]['None'].keys(), fontsize=16)
        plt.yticks(fontsize=18)
        plt.ylim(0, 100)
        if not i:
            plt.ylabel('WinRate / ASR (%)', fontsize=20)
            plt.legend(loc='upper left', framealpha=0.5, fontsize=18)
        plt.title(model_name, fontsize=24)
    plt.savefig(save_dir + f'/main_more.pdf', dpi=1000, bbox_inches='tight')
    plt.clf()

def main_instruct():
    # python log.py -m huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00 huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00 huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00_dpo_NaiveCompletion_2024-07-06-07-42-23 mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11 mistralai/Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion_2024-07-20-05-46-17 mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11_dpo_NaiveCompletion_2024-08-13-17-46-51 meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02 meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_NaiveCompletion_2024-08-09-12-55-56 meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02_dpo_NaiveCompletion_2024-08-09-21-28-53
    main_results = {
        'Mistral-7B-Instruct': { # 'Mistral-7B-v0.1'
            'None': {
                'AlpacaEval2\nWinRate (↑)': 67.01,
                'Max ASR (↓)\nOpt.-Free': 59,
                #'AdvPrompter\nASR (↓)': 72,
                'Max ASR (↓)\nOpt.-Based': 89,#99,
                },
            'StruQ': {
                'AlpacaEval2\nWinRate (↑)': 70.73,
                'Max ASR (↓)\nOptimization-Free': 2,
                #'AdvPrompter\nASR (↓)': 7,
                'Max ASR (↓)\nOptimization-Based': 34, #56,
                },
            'SecAlign': {
                'AlpacaEval2\nWinRate (↑)': 69.22,
                'Max ASR (↓)\nOptimization-Free': 0,
                #'AdvPrompter\nASR (↓)': 0,
                'Max ASR (↓)\nOptimization-Based': 1, #2,
                },
            },
        'Llama3-8B-Instruct': { # Meta-Llama-3-8B
            'None': {
                'AlpacaEval2\nWinRate (↑)': 85.39,
                'Max ASR (↓)\nOpt.-Free': 52,
                #'AdvPrompter\nASR (↓)': 95,
                'Max ASR (↓)\nOpt.-Based': 98, #95, #89,
                },
            'StruQ': {
                'AlpacaEval2\nWinRate (↑)': 80.79,
                'Max ASR (↓)\nOptimization-Free': 0,
                #'AdvPrompter\nASR (↓)': 18,
                'Max ASR (↓)\nOptimization-Based': 62, #33,
                },
            'SecAlign': {
                'AlpacaEval2\nWinRate (↑)': 85.88,
                'Max ASR (↓)\nOptimization-Free': 0,
                #'AdvPrompter\nASR (↓)': 0,
                'Max ASR (↓)\nOptimization-Based': 1, #11,
                },
            }
        }

    width = 0.4
    #r1 = np.arange(len(main_results['Llama-7B']['None']))
    r1 = np.array([0, 1.2, 2.2])
    barwidth = len(r1) * 0.4 + width
    r1 = r1 * barwidth
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r = [r1, r2, r3]
    
    #plt.figure(figsize=(16,6))
    plt.figure(figsize=(8,6))
    for i, model_name in enumerate(main_results.keys()):
        #plt.subplot(1, 2, i+1)
        for j, defense in enumerate(main_results[model_name]):
            plt.bar(r[j], main_results[model_name][defense].values(), color=colors[defense], label=defense, width=width)
            for k, v in enumerate(main_results[model_name][defense].values()): 
                if v == 0: plt.text(r[j][k], v + 1, str(v) + '%', ha='center')
                elif v < 1: plt.text(r[j][k], v + 1, '.5%', ha='center')
                elif v == 1: plt.text(r[j][k], v + 1, '1%', ha='center')
                elif v == 2: plt.text(r[j][k], v + 1, '2%', ha='center')
        plt.gca().axvline(x=(r1[0] + r1[1])/2 + width, color='grey', linewidth=1)
        plt.xticks([k + width for k in r1], main_results[list(main_results.keys())[0]]['None'].keys(), fontsize=16)
        plt.yticks(fontsize=18)
        plt.ylim(0, 100)
        #if not i:
        plt.ylabel('WinRate / ASR (%)', fontsize=20)
        if 'Llama3' not in model_name: plt.legend(loc='upper left', framealpha=0.5, fontsize=18)

        plt.title(model_name, fontsize=24)
        plt.savefig(save_dir + f'/main_instruct_{model_name}.pdf', dpi=1000, bbox_inches='tight')
        plt.clf()

def main_instruct_intro():
    main_results = {
        'Llama3-8B-Instruct': { # Meta-Llama-3-8B
            'No defense': {
                'AlpacaEval2\n(↑ for better utility)': 85.39,
                'Max Attack Success Rate\n(↓ for better security)': 98,
                },
            'SOTA prompting-based defense': {
                'AlpacaEval2\n(↑ for better utility)': 73.71, #76.73,
                'Max Attack Success Rate\n(↓ for better security)': 96,#unknown #50 #90,
                },
            'SOTA fine-tuning-based defense': {
                'AlpacaEval2\n(↑ for better utility)': 80.79,
                'Max Attack Success Rate\n(↓ for better security)': 62,
                },
            'SecAlign fine-tuning-based defense': {
                'AlpacaEval2\n(↑ for better utility)': 85.88,
                'Max Attack Success Rate\n(↓ for better security)': 1,
                },
            }
        }

    width = 0.4
    r1 = np.array([0, 2.1])
    barwidth = len(r1) * 0.4 + width
    r1 = r1 * barwidth
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]
    r = [r1, r2, r3, r4]
    plt.tick_params(bottom=False)
    plt.axes().xaxis.set_ticks_position('none') 
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.tick_params(axis='x', which='both', length=0)
    #plt.figure(figsize=(8.5, 10))
    plt.figure(figsize=(8.5, 6))
    for i, model_name in enumerate(main_results.keys()):
        plt.subplot(1, 1, i+1)
        for j, defense in enumerate(main_results[model_name]):
            plt.bar(r[j], main_results[model_name][defense].values(), color=colors[defense], label=defense, width=width)
            for k, v in enumerate(main_results[model_name][defense].values()): 
                if v == 0: plt.text(r[j][k], v + 1, str(v) + '%', ha='center')
                elif v < 1: plt.text(r[j][k], v + 1, '.5%', ha='center')
                elif v == 1: plt.text(r[j][k], v + 1, '1%', ha='center')
        plt.gca().axvline(x=(r1[0] + r1[1])/2 + width * 1.5, color='grey', linewidth=1)
        plt.xticks([k + width*0.5 for k in r1] + [k + width*1.5 for k in r1] + [k + width*2.5 for k in r1]
                   , list(main_results[list(main_results.keys())[0]]['SOTA prompting-based defense'].keys()) + ['', '', '', ''], fontsize=22)
        plt.yticks(fontsize=22)
        #plt.ylim(0, 130)
        plt.ylim(0, 175)
        if not i:
            plt.legend(loc='upper center', framealpha=0.5, fontsize=22)
        plt.title(model_name, fontsize=22)
    plt.savefig(save_dir + f'/main_instruct_intro.pdf', dpi=1000, bbox_inches='tight')
    plt.savefig(save_dir + f'/main_instruct_intro.png', dpi=300, bbox_inches='tight')
    plt.clf()


def main_struq():
    # python log.py -m huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00 huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00 huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00_dpo_NaiveCompletion_2024-07-06-07-42-23 mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11 mistralai/Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion_2024-07-20-05-46-17 mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11_dpo_NaiveCompletion_2024-08-13-17-46-51 meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02 meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_NaiveCompletion_2024-08-09-12-55-56 meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02_dpo_NaiveCompletion_2024-08-09-21-28-53
    main_results = {
        'Llama-7B': { # 'llama-7b'
            'None': {
                'AlpacaEval2\nWinRate (↑)': 55.46,
                'Max ASR (↓)\nOpt.-Free': 75,
                #'AdvPrompter\nASR (↓)': 60,
                'Max ASR (↓)\nOpt.-Based': 97,
                },
            'StruQ': {
                'AlpacaEval2\nWinRate (↑)': 54.55,
                'Max ASR (↓)\nOptimization-Free': 0.5,
                #'AdvPrompter\nASR (↓)': 4,
                'Max ASR (↓)\nOptimization-Based': 60, #58,
                },
            },
        'Mistral-7B': { # 'Mistral-7B-v0.1'
            'None': {
                'AlpacaEval2\nWinRate (↑)': 72.21,
                'Max ASR (↓)\nOpt.-Free': 89,
                #'AdvPrompter\nASR (↓)': 72,
                'Max ASR (↓)\nOpt.-Based': 95,#99,
                },
            'StruQ': {
                'AlpacaEval2\nWinRate (↑)': 72.17,
                'Max ASR (↓)\nOptimization-Free': 4,
                #'AdvPrompter\nASR (↓)': 7,
                'Max ASR (↓)\nOptimization-Based': 41, #56,
                },
            },
        'Llama3-8B': { # Meta-Llama-3-8B
            'None': {
                'AlpacaEval2\nWinRate (↑)': 69.47,
                'Max ASR (↓)\nOpt.-Free': 90,
                #'AdvPrompter\nASR (↓)': 95,
                'Max ASR (↓)\nOpt.-Based': 98, #95, #89,
                },
            'StruQ': {
                'AlpacaEval2\nWinRate (↑)': 68.77,
                'Max ASR (↓)\nOptimization-Free': 0,
                #'AdvPrompter\nASR (↓)': 18,
                'Max ASR (↓)\nOptimization-Based': 43, #33,
                },
            }
        }

    width = 0.4
    #r1 = np.arange(len(main_results['Llama-7B']['None']))
    r1 = np.array([0, 1, 2]) #np.array([0, 1.2, 2.2])
    barwidth = len(r1) * 0.4 + width
    r1 = r1 * barwidth
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r = [r1, r2]#, r3]
    
    plt.figure(figsize=(18,6))
    for i, model_name in enumerate(main_results.keys()):
        plt.subplot(1, 3, i+1)
        for j, defense in enumerate(main_results[model_name]):
            plt.bar(r[j], main_results[model_name][defense].values(), color=colors[defense], label=defense, width=width)
            for k, v in enumerate(main_results[model_name][defense].values()): 
                if v == 0: plt.text(r[j][k], v + 1, str(v) + '%', ha='center')
                elif v < 1: plt.text(r[j][k], v + 1, '.5%', ha='center')
        plt.gca().axvline(x=(r1[0] + r1[1])/2 + width/2, color='grey', linewidth=1)
        plt.xticks([k + width/2 for k in r1], main_results['Llama-7B']['None'].keys(), fontsize=16)
        plt.yticks(fontsize=18)
        plt.ylim(0, 100)
        if model_name == "Llama-7B":
            plt.ylabel('WinRate / ASR (%)', fontsize=20)
            plt.legend(loc='upper center', framealpha=0.5, fontsize=18)
        plt.title(model_name, fontsize=24)
    plt.savefig(save_dir + f'/main_struq.pdf', dpi=1000, bbox_inches='tight')
    plt.clf()


def prop():
    plt.rcParams['figure.figsize'] = [5.3, 3.2]#[6.4, 3.8]
    proportions = [0.2, 0.4, 0.6, 0.8, 1]
    plt.plot(proportions, [51.41, 55.16, 55.14, 55.10, 54.55], linewidth=2, color=colors['StruQ'], linestyle='dotted', label='StruQ (WinRate)')
    plt.plot(proportions, [72, 71, 71, 60, 58], linewidth=2, color=colors['StruQ'], label='StruQ (GCG ASR)')
    plt.plot(proportions, [50.44, 53.33, 57.85, 56.39, 56.06], linewidth=2, color=colors['SecAlign'], linestyle='dotted', label='SecAlign (WinRate)')
    plt.plot(proportions, [30, 26, 29, 23, 15], linewidth=2, color=colors['SecAlign'], label='SecAlign (GCG ASR)')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Ratio of the training data used', fontsize=18)
    plt.ylabel('WinRate / ASR (%)', fontsize=18)
    plt.legend(loc='center right', framealpha=0.3, fontsize=14)
    plt.savefig(save_dir + '/prop.pdf', dpi=1000, bbox_inches='tight')
    plt.clf()


def lr():
    plt.rcParams['figure.figsize'] = [5.3, 3.2]#[6.4, 3.8]
    lrs = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    plt.plot(lrs, [54.20, 55.50, 55.92, 56.09, 55.84, 55.55, 55.59, 56.06, 55.31, 56.16, 56.33], linewidth=2, color=colors['SecAlign'], linestyle='dotted', label='SecAlign (WinRate)')
    plt.plot(lrs, [44, 34, 31, 27, 22, 17, 19, 15, 16, 20, 24], linewidth=2, color=colors['SecAlign'], label='SecAlign (GCG ASR)')
    plt.plot(lrs, [54.55 for x in lrs], linewidth=2, color=colors['StruQ'], linestyle='dotted', label='StruQ (WinRate)')
    plt.plot(lrs, [58 for x in lrs], linewidth=2, color=colors['StruQ'], label='StruQ (GCG ASR)')
    err_utility = np.std([56.64, 56.78, 56.57, 57.26, 56.47]) / np.sqrt(5)
    err_security = np.std([18, 21, 20, 17, 21]) / np.sqrt(5)
    width = 0.15
    plt.gca().add_patch(Rectangle((20-width, 56.06 - err_utility), 2 * width, 2 * err_utility, color=colors['SecAlign']))
    plt.gca().add_patch(Rectangle((20-width, 15 - err_security), 2 * width, 2 * err_security, color=colors['SecAlign']))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('SecAlign DPO learning rate (e-5)', fontsize=18)
    plt.ylabel('WinRate / ASR (%)', fontsize=18)
    plt.legend(fontsize=16)
    plt.savefig(save_dir + '/lr.pdf', dpi=1000, bbox_inches='tight')
    plt.clf()


def intro():
    plt.rcParams['figure.figsize'] = [6.4, 10]
    intro_results = np.array([95, 41, 1])
    width = 0.5
    r = np.arange(3) * width
    plt.bar(r, intro_results, width=width, color=[colors['None'], colors['StruQ'], colors['SecAlign']])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.box(False) 
    for i, d in enumerate(intro_results): plt.text(r[i], d + 2, str(d) + '%', ha='center', rotation='vertical', fontsize='xx-large')
    path = save_dir + '/intro.pdf'
    plt.savefig(path, dpi=1000, bbox_inches='tight')
    plt.clf()
    

def loss():
    plt.rcParams['figure.figsize'] = [6.4, 3.2]
    gcg_logs = {
        'None': 'meta-llama/Meta-Llama-3-8B-Instruct-log/gcg/len20_500step_bs512_seed0_l50_t1.0_static_k256',
        'StruQ': 'meta-llama/Meta-Llama-3-8B-Instruct_Meta-Llama-3-8B-Instruct_NaiveCompletion_2024-11-12-17-58-38/gcg/len20_500step_bs512_seed0_l50_t1.0_static_k256',
        'SecAlign': 'meta-llama/Meta-Llama-3-8B-Instruct_dpo_NaiveCompletion_2024-11-12-17-59-06/gcg/len20_500step_bs512_seed0_l50_t1.0_static_k256',
    }

    def extract_log_dict(log_file):
        loss = []
        with open(log_file, 'r') as json_file: json_list = list(json_file)
        for json_str in json_list:
            loss.append(json.loads(json_str)['loss'])
        while len(loss) != 26: loss.append(loss[-1])
        return loss
    
    exists = []
    for defense in gcg_logs:
        losses = []
        for i in range(208): 
            try: losses.append(extract_log_dict(gcg_logs[defense] + f'/{i}.jsonl')); exists.append(i)
            except FileNotFoundError: pass #print(i)
        losses = np.array(losses)
        x = np.arange(losses.shape[1]) * 20
        mean_values = np.mean(losses, axis=0)
        std_values = np.std(losses, axis=0)
        plt.plot(x, mean_values, color=colors[defense], linewidth=2, label=defense)
        plt.fill_between(x, np.subtract(mean_values, std_values), np.add(mean_values, std_values), alpha=0.2, color=colors[defense])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('GCG step(s)', fontsize=18)
    plt.ylabel('GCG Attack Loss', fontsize=18)
    plt.legend(fontsize=16, loc='lower right', framealpha=0.5)
    plt.savefig(save_dir + '/loss.pdf', dpi=1000, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Plot a figure')
    parser.add_argument('-f', '--fig', type=str)
    eval(parser.parse_args().fig)()