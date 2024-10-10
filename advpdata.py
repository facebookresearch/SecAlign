# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
from config import TEST_INJECTED_PROMPT, TEST_INJECTED_WORD, DELIMITERS
from struq import jload
import argparse
from copy import deepcopy
import numpy as np

parser = argparse.ArgumentParser(prog='Generating Advprompter Data')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--no_test_inject', default=False, action='store_true')
parser.add_argument("--max_train_samples", type=int, default=300)
args = parser.parse_args()

target_dataset_path = {
    False: 'advprompter/data/prompt_injections/dataset/test.csv',
    True: f'advprompter/data/prompt_injections/dataset/train_{args.max_train_samples}.csv'
}[args.train]

data_path = {
    False: 'data/davinci_003_outputs.json',
    True: 'data/alpaca_data_cleaned.json'
}[args.train]


prompt_format = DELIMITERS['SpclSpclSpcl'][0] + "\n{instruction}\n\n" + DELIMITERS['SpclSpclSpcl'][1] + "\n{input}"
data = jload(data_path)
cnt = 0
with open(target_dataset_path, "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['instruct', 'target'])
    for d in data:
        if d['input'] == '': continue
        current_sample = deepcopy(d)
        if not args.train: 
            if not args.no_test_inject: 
                if current_sample['input'][-1] not in ['.', '!', '?']: current_sample['input'] += '.'
                current_sample['input'] += ' ' + TEST_INJECTED_PROMPT
            writer.writerow([prompt_format.format_map(current_sample), TEST_INJECTED_WORD + '!'])
        else:
            is_question = current_sample['input'][-1] == '?'
            if current_sample['input'][-1] not in ['.', '!', '?']: current_sample['input'] += '.'
            injected_sample = np.random.choice(data) 
            injected_prompt = injected_sample['instruction'] + ' ' + injected_sample['input']
            if is_question: injected_prompt = 'Answer the following question. ' + injected_prompt
            current_sample['input'] += ' ' + injected_prompt.capitalize()
            writer.writerow([prompt_format.format_map(current_sample), injected_sample['output']])
        cnt += 1
        if cnt >= args.max_train_samples: break
