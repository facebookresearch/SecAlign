# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

if [[ $2 == 'run' ]]; then
    nohup python -u run.py --do_sft --sft_attack SpclSpclSpcl_NaiveCompletion --do_test -m $1 > struq.log 2>&1 &
else
    python run.py --do_sft --sft_attack SpclSpclSpcl_NaiveCompletion --do_test -m $1 --print_cmd
fi