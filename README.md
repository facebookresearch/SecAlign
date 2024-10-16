# Decription
+ The code is the official implementation of the SecAlign paper: [Aligning LLMs to Be Robust Against Prompt Injection](https://arxiv.org/abs/2410.05451)
+ Training SecAlign / [StruQ](https://github.com/Sizhe-Chen/StruQ) LLMs requires 4 80G A100s. Testing utility and manual attacks requires 1 16G GPU. Testing [GCG](https://github.com/llm-attacks/llm-attacks) requires 1 80G A100. Testing [AdvPrompter](https://github.com/facebookresearch/advprompter) requires 2 80G A100s.

# Environment
### Packages
+ clone this repo and ```cd SecAlign```
+ create the conda env by running ```conda create -n secalign python==3.10```. If you use another env name, specify it in the ```-e``` in ```run.py```
+ install dependencies by running ```pip install -r requirements.txt```
### Data
+ data/[alpaca_data_clean.json](https://github.com/gururise/AlpacaDataCleaned/blob/main/alpaca_data_cleaned.json): training set
+ data/[alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json): reference set for training-time completion attacks
+ data/[davinci_003_outputs.json](https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/main/davinci_003_outputs.json): testing set for utility and security
+ data/[openai_configs.yaml](https://github.com/tatsu-lab/alpaca_eval/tree/main/client_configs): configure your openai key for evaluating utility
+ data/[cyberseceval3_prompt_injection.json](https://github.com/meta-llama/PurpleLlama/blob/main/CybersecurityBenchmarks/datasets/prompt_injection/prompt_injection.json): testing set for [CyberSecEval3](https://github.com/meta-llama/PurpleLlama/blob/main/CybersecurityBenchmarks)
+ Advprompter testing data: ```python advpdata.py```, save to ```advprompter/data/prompt_injections/dataset/test.csv```
+ [optional] Study Data efficiency. Set ```--data data/alpaca_data_cleaned_0.2.json``` in ```run.py``` to launch ablation. This script generate different proportions of ```data/alpaca_data_cleaned.json```, and always save a sandwich defended (strongest prompting defense) version of it for testing GCG.
> python propdata.py
+ [optional] Study injected word: change TEST_INJECTED_WORDS in ```config.py```
+ [optional] Tune lr: change LR_CONFIG in ```run.py```
+ [optional] Draw figures: ```[figure_name]``` could be ['intro', 'main', 'motivation', loss', 'lr']
> python figure.py -f [figure_name]
### Models [optional]
##### Undefended
+ huggyllama/[llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00](https://dl.fbaipublicfiles.com/SecAlign/huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00.zip)
+ mistralai/[Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11](https://dl.fbaipublicfiles.com/SecAlign/mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11.zip)
+ meta-llama/[Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02](https://dl.fbaipublicfiles.com/SecAlign/meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02.zip)
##### StruQ
+ huggyllama/[llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00](https://dl.fbaipublicfiles.com/SecAlign/huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00.zip)
+ mistralai/[Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion_2024-07-20-05-46-17](https://dl.fbaipublicfiles.com/SecAlign/mistralai/Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion_2024-07-20-05-46-17.zip)
+ meta-llama/[Meta-Llama-3-8B_SpclSpclSpcl_NaiveCompletion_2024-08-09-12-55-56](https://dl.fbaipublicfiles.com/SecAlign/meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_NaiveCompletion_2024-08-09-12-55-56.zip)
##### SecAlign
+ huggyllama/[llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00_dpo_NaiveCompletion_2024-07-06-07-42-23](https://dl.fbaipublicfiles.com/SecAlign/huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00_dpo_NaiveCompletion_2024-07-06-07-42-23.zip)
+ mistralai/[Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11_dpo_NaiveCompletion_2024-08-13-17-46-51](https://dl.fbaipublicfiles.com/SecAlign/mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11_dpo_NaiveCompletion_2024-08-13-17-46-51.zip)
+ meta-llama/[Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02_dpo_NaiveCompletion_2024-08-09-21-28-53](https://dl.fbaipublicfiles.com/SecAlign/meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02_dpo_NaiveCompletion_2024-08-09-21-28-53.zip)

# Train
+ Specify your slurm partition and other configurations in ```run.py```, which performs training and testing by generating slurm scripts, run them, and delete them. It uses an additional thread from ```nohup``` to moniter the training, and automatically tests after the training finishes if ```--do_test``` is specified, with the default ```--test_attack none ignore completion_real completion_realcmb advp gcg```
+ SecAlign requires a SFT model to do preference optimization. Run SFT beforehand, or download corresponding SFT models.
+ See ```secalign_script.sh``` for commands to train Undefended/StruQ/SecAlign models for ```huggyllama/llama-7b```, ```mistralai/Mistral-7B-v0.1```, and ```meta-llama/Meta-Llama-3-8B```.

# Test
+ All logs on training / utility / security are saved to the model path, with a ```summary.tsv```.
+ Parallel test: none for utility, prompting defense could be ['none', 'sandwich', 'instructional', 'reminder', 'isolation', 'incontext']
> python run.py --do_test --test_attack none ignore completion_real completion_realcmb gcg advp -m [model_path] -d [defense]
+ Sequential test
> python test.py -a none ignore completion_real completion_realcmb -m [model_path] \
> python test_gcg.py -m [model_path] --sample_ids 0 1 206 207 \
> python run_advp_gcg.py -a gcg -c log -m [model_path]

# Code Acknowledgements
The majority of SecAlign and the included [StruQ](https://github.com/Sizhe-Chen/StruQ) and [AdvPrompter](https://github.com/facebookresearch/advprompter) are licensed under CC-BY-NC, however portions of the project are available under separate license terms: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) is licensed Apache 2.0; [LLM Attacks](https://github.com/llm-attacks/llm-attacks) is licensed MIT.

Code under `gcg/` is adapted from [LLM Attacks](https://github.com/llm-attacks/llm-attacks). Code under `advprompter/` is adapted from https://github.com/facebookresearch/advprompter.
