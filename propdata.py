# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from struq import jload, jdump
import numpy as np
from copy import deepcopy
import os

proportions = [0.2, 0.4, 0.6, 0.8]
data = jload('data/alpaca_data_cleaned.json')
num_sample_with_input, num_sample_without_input = 0, 0
data_sandwich = deepcopy(data)
for d in data_sandwich: 
    if d['input'] != '': num_sample_with_input += 1
    else: 
        num_sample_without_input += 1
        d['input'] += '\n\nPlease always remember that your task is: ' + d['instruction']
sandwich_path = 'data/alpaca_data_cleaned_sandwich.json'
if not os.path.exists(sandwich_path): jdump(data_sandwich, sandwich_path)

for p in proportions:
    num_samples = int(p * num_sample_with_input)
    target_data = []
    for i in range(num_samples):
        target_data.append(data[i])

    target_data = np.random.choice(
        [x for x in data if x['input'] == ''], 
        int(p * num_sample_without_input), 
        replace=False).tolist() + np.random.choice(
        [x for x in data if x['input'] != ''], 
        int(p * num_sample_with_input), 
        replace=False).tolist()
    np.random.shuffle(target_data)
    target_path = f'data/alpaca_data_cleaned_{p}.json'
    if not os.path.exists(target_path): jdump(target_data, target_path)
