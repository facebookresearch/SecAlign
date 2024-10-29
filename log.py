# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import argparse

def log_gcg(model_path):
    log_path = model_path + '/gcg/len20_500step_bs512_seed0_l50_t1.0_static_k256'
    begin_with, in_response = 0, 0
    logs = os.listdir(log_path)
    num_invalid = 0
    for gcg_log in logs:
        with open(os.path.join(log_path, gcg_log), 'r') as f:
            txt = f.read()
            if 'begin_with": true' in txt: begin_with += 1
            elif txt.count('{"loss":') != 26: num_invalid += 1
            if 'in_response": true' in txt: in_response += 1
    begin_with /= len(logs)-num_invalid
    in_response /= len(logs)-num_invalid
    print(f"\n{log_path} Begin_with {begin_with: <30} In_response {in_response: <30} Length {len(logs): <10}/{len(logs)-num_invalid: <10}")
    with open(model_path + '/summary.tsv', "a") as outfile: outfile.write(f"gcg\t{in_response}\t{begin_with}\tUnknown_{len(logs)-num_invalid}\n")

def log_adv(model_path):
    for file in os.listdir(model_path):
        if 'advp' not in file: continue
        log_path = model_path + '/' + file
        with open(log_path, 'r') as f:
            txt = f.read()
            if 'in-response rate @ 1: ' not in txt or 'begin-with rate @ 1: ' not in txt: continue
            begin_with = max([float(x) for x in re.findall(r"begin-with rate @ 1: (.*?)\n", txt)])
            in_response = max([float(x) for x in re.findall(r"in-response rate @ 1: (.*?)\n", txt)])
            print(f"\n{log_path} Begin_with {begin_with: <30} In_response {in_response: <30} Length {txt.count('begin-with rate @ 1: '): <10}")
            with open(model_path + '/summary.tsv', "a") as outfile: outfile.write(f"advp\t{in_response}\t{begin_with}\tUnknown_{os.path.basename(log_path)}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Log GCG and AdvPrompter attack results to model_path/summary.tsv')
    parser.add_argument('-m', '--model_path', nargs="+", type=str, help='Path of the model to test')
    args = parser.parse_args()
    for model_path in args.model_path: 
        summary_path = model_path + '/summary.tsv'
        if not os.path.exists(summary_path):
            with open(summary_path, "w") as outfile: 
                outfile.write("attack\tin-response\tbegin-with\tdefense\n")
        log_gcg(model_path)
        log_adv(model_path)