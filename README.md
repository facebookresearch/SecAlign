# Decription
+ The code is the official implementation of the SecAlign paper: [Aligning LLMs to Be Robust Against Prompt Injection](https://arxiv.org/abs/2410.05451)
+ Training SecAlign / [StruQ](https://github.com/Sizhe-Chen/StruQ) LLMs requires 4 80G A100s. Testing utility and manual attacks requires 1 16G GPU. Testing [GCG](https://github.com/llm-attacks/llm-attacks) requires 1 80G A100. Testing [AdvPrompter](https://github.com/facebookresearch/advprompter) requires 2 80G A100s.

# Environment
+ Install environment dependencies
> git clone https://github.com/facebookresearch/SecAlign \
> cd SecAlign \
> conda create -n secalign python==3.10
+ Install package dependencies
> pip install -r requirements.txt
+ Download data dependencies
> python setup.py
+ Configure openai dependencies for utility evaluation: create ```data/openai_configs.yaml``` following ```data/openai_configs_examle.yaml```
+ [optional] Download trained models to play. If you want to run SecAlign but skip the required SFT computation, download the SFT undefended models! This command downloads 9 Undefended / StruQ / SecAlign models (3 architectures).
> python setup.py --model
+ [optional] Automatic and efficient testing by specifying your training/testing slurm configurations in the ```slurm_prefix``` variables in ```run.py```, which generates slurm scripts, run them, and delete them. It supports an additional thread from ```nohup``` to moniter the training, and automatically tests after the training finishes if ```--do_test``` is specified

# Train
+ SFT (Undefended): get the slurm and python commands, and run by yourself. The ```[model_path]``` below stands for the huggingface model ID (supports ```huggyllama/llama-7b```, ```mistralai/Mistral-7B-v0.1```, and ```meta-llama/Meta-Llama-3-8B```) or your local model path.
> bash scripts/undefended.sh -m [model_path]
+ SFT (Undefended): run the training, and test immediately after the training simultaneously on multiple GPUs, with the default ```--test_attack none ignore completion_real completion_realcmb gcg``` (```none``` for utility)
> bash scripts/undefended.sh -m [model_path] run
+ SFT (StruQ)
> bash scripts/struq.sh -m [model_path] \
> bash scripts/struq.sh -m [model_path] run
+ Align (SecAlign): requires a SFT model, and specify its path in ```[model_path]```
> bash scripts/secalign.sh -m [model_path] \
> bash scripts/secalign.sh -m [model_path] run

# Test
+ All logs on training, utility evaluation, and security evaluation are saved to ```[model_path]/summary.tsv``` if you use ```bash [script_path] -m [model_path] run```
+ Test: get the slurm and python commands, and run by yourself. The default ```--test_attack none ignore completion_real completion_realcmb gcg``` (```none``` for utility)
> bash scripts/test.sh -m [model_path]
+ Test: run all tests simultaneously on multiple GPUs
> bash scripts/test.sh -m [model_path] run
+ Test: customize the ```--test_attack```, prompting-based ```--defense```, and testing ```--data_path```, and then run all tests simultaneously on multiple GPUs (You may add ```--print_cmd``` after the below command if you want to get the slurm and python commands, and run by yourself). The ```--defense``` could be ['none', 'sandwich', 'instructional', 'reminder', 'isolation', 'incontext'], and ```--test_attack``` could be ['naive', 'ignore', 'escape_deletion', 'escape_separation', 'completion_other', 'completion_othercmb', 'completion_real', 'completion_realcmb', 'completion_close_2hash', 'completion_close_1hash', 'completion_close_0hash', 'completion_close_upper', 'completion_close_title', 'completion_close_nospace', 'completion_close_nocolon', 'completion_close_typo', 'completion_close_similar', 'hackaprompt', 'gcg', 'advp']
> python run.py --do_test --test_attack [test_attack1] [test_attack2] [test_attack3] -m [model_path1] [model_path2] [model_path3] -d [defense] --data_path [data_path]
+ Log: log the GCG and AdvPrompter testing results to ```[model_path]/summary.tsv```. To support this automatic logging, AdvPrompter has to be run through ```bash``` or ```python run.py```, which produces a ```adv_jobID.out``` in ```[model_path]```
> python log.py -m [model_path]

# Misc
+ Tune lr: change LR_CONFIG in ```run.py```
+ Study data efficiency: set ```--data data/alpaca_data_cleaned_0.2.json``` in ```run.py```
+ Study injected word: change TEST_INJECTED_WORDS in ```config.py```
+ Draw figures: run ```python figure.py -f [figure_name]```, where ```[figure_name]``` could be ['intro', 'main', 'motivation', loss', 'lr']

# Code Acknowledgements
The majority of SecAlign and the included [StruQ](https://github.com/Sizhe-Chen/StruQ) and [AdvPrompter](https://github.com/facebookresearch/advprompter) are licensed under CC-BY-NC, however portions of the project are available under separate license terms: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) is licensed Apache 2.0; [LLM Attacks](https://github.com/llm-attacks/llm-attacks) is licensed MIT. Code under `gcg/` is adapted from [LLM Attacks](https://github.com/llm-attacks/llm-attacks). Code under `advprompter/` is adapted from [AdvPrompter](https://github.com/facebookresearch/advprompter).