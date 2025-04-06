# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

if [[ $2 == 'run' ]]; then
    python run.py --do_test --test_attack none ignore completion_real completion_realcmb gcg advp -m $1
else
    python run.py --do_test --test_attack none ignore completion_real completion_realcmb gcg advp -m $1 --print_cmd
fi