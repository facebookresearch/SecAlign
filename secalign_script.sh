# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

### Undefended
nohup python -u run.py --do_sft --sft_attack SpclSpclSpcl_None --do_test -m huggyllama/llama-7b > undefended_llama.log 2>&1 &
nohup python -u run.py --do_sft --sft_attack SpclSpclSpcl_None --do_test -m mistralai/Mistral-7B-v0.1 > undefended_mistral.log 2>&1 &
nohup python -u run.py --do_sft --sft_attack SpclSpclSpcl_None --do_test -m meta-llama/Meta-Llama-3-8B > undefended_llamathree.log 2>&1 &

### StruQ
nohup python -u run.py --do_sft --sft_attack SpclSpclSpcl_NaiveCompletion --do_test -m huggyllama/llama-7b > struq_llama.log 2>&1 &
nohup python -u run.py --do_sft --sft_attack SpclSpclSpcl_NaiveCompletion --do_test -m mistralai/Mistral-7B-v0.1 > struq_mistral.log 2>&1 &
nohup python -u run.py --do_sft --sft_attack SpclSpclSpcl_NaiveCompletion --do_test -m meta-llama/Meta-Llama-3-8B > struq_llamathree.log 2>&1 &

### SecAlign (Requiring a SFT model to do preference optimization. Run SFT training beforehand, or download corresponding SFT models)
nohup python -u run.py --do_align --alignment dpo --align_attack NaiveCompletion --do_test -m huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00 > secalign_llama.log 2>&1 &
nohup python -u run.py --do_align --alignment dpo --align_attack NaiveCompletion --do_test -m mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11 > secalign_mistral.log 2>&1 &
nohup python -u run.py --do_align --alignment dpo --align_attack NaiveCompletion --do_test -m meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02 > secalign_llamathree.log 2>&1 &