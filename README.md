# SecAlign: Defending Against Prompt Injection with Preference Optimization
[Sizhe Chen](https://sizhe-chen.github.io), [Arman Zharmagambetov](https://arman-z.github.io), [Saeed Mahloujifar](https://smahloujifar.github.io), [Kamalika Chaudhuri](https://cseweb.ucsd.edu/~kamalika), [David Wagner](https://people.eecs.berkeley.edu/~daw), [Chuan Guo](https://sites.google.com/view/chuanguo)

[![](https://img.shields.io/badge/CCS'25-a8c66c)](https://arxiv.org/pdf/2410.05451) [![](https://img.shields.io/badge/Website-097770)](https://sizhe-chen.github.io/SecAlign-Website) [![](https://img.shields.io/badge/Poster-1b6535)](https://drive.google.com/file/d/1-HFnET2azKniaS4k5dvgVwoRLa4Eg584/view?usp=sharing) [![](https://img.shields.io/badge/Talk-edca82)](https://docs.google.com/document/d/1pip5y_HGU4qjN0K6NEFuI379RPdL9T6o/edit?usp=sharing)  [![](https://img.shields.io/badge/Slides-f47a60)](https://drive.google.com/file/d/1baUbgFMILhPWBeGrm67XXy_H-jO7raRa/view?usp=sharing)

Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the Internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be injected into external data sources to override the system's intended instruction and instead execute a malicious instruction. To mitigate this vulnerability, we propose a new defense called SecAlign based on the technique of preference optimization. Our defense first constructs a preference dataset with prompt-injected inputs, secure outputs (ones that respond to the legitimate instruction), and insecure outputs (ones that respond to the injection). We then perform preference optimization on this dataset to teach the LLM to prefer the secure output over the insecure one. This provides the first known method that reduces the success rates of various prompt injections to around 0%, even against attacks much more sophisticated than ones seen during training. This indicates our defense generalizes well against unknown and yet-to-come attacks. Also, our defended models are still practical with similar utility to the one before our defensive training.

SecAlign Overview           |  SecAlign Main Results
:-------------------------:|:-------------------------:
![SecAlign Overview](figures/teaser.png)  |  ![SecAlign Main Results](figures/main_instruct_intro.png)


# Environment
+ Training SecAlign / [StruQ](https://github.com/Sizhe-Chen/StruQ) LLMs requires 4 80G A100s. Testing utility and manual attacks requires 1 16G GPU. Testing [GCG](https://github.com/llm-attacks/llm-attacks) requires 1 80G A100. Testing [AdvPrompter](https://github.com/facebookresearch/advprompter) requires 2 80G A100s.
+ Install environment dependencies
> git clone https://github.com/facebookresearch/SecAlign \
> cd SecAlign \
> conda create -n secalign python==3.10
+ Install package dependencies
> pip install -r requirements.txt
+ Download data dependencies
> python setup.py
+ Configure openai dependencies for utility evaluation: create ```data/openai_configs.yaml``` following ```data/openai_configs_examle.yaml```
+ [optional] Play with SecAlign Instruct models. Run ```python setup.py --instruct``` to download SecAlign Instruct LoRA adapters (0.2G). 
+ [optional] Play with SecAlign Alpaca models. Run ```python setup.py --alpaca``` to download SFTed (on [alpaca_data_cleaned.json](https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json)) Llama-7B (26G), Mistral-7B (27G), Llama3-8B (30G) and the corresponding SecAlign LoRA adapters (0.4G).
+ [optional] Automatic and efficient testing by specifying your training/testing slurm configurations in the ```slurm_prefix``` variables in ```run.py```, which generates slurm scripts, run them, and delete them. It supports an additional thread from ```nohup``` to moniter the training, and automatically tests after the training finishes if ```--do_test``` is specified

# SecAlign Preference Optimization
+ Get the slurm and python commands, and run by yourself. The ```[model_path]``` below stands for the huggingface Instruct model ID (currently supporting ```mistralai/Mistral-7B-v0.1-Instruct``` and ```meta-llama/Meta-Llama-3-8B-Instruct```) or your local SFTed (see next section) model path.
> bash scripts/secalign.sh [model_path]
+ Run the training, and test immediately after the training simultaneously on multiple GPUs, with the default ```--test_attack none ignore completion_real completion_realcmb gcg advp``` (```none``` for utility)
> bash scripts/undefended.sh [model_path] run
+ To tune the learning rate, change dpo_lr in LR_CONFIG in ```run.py```

# SFT (Supervised Fine-Tuning)
+ SecAlign starts on an SFTed model. If you do not want to use public SFTed models (as in the last section), you can SFT your own model from a base model. 
+ The below command SFTs on [alpaca_data_cleaned.json](https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json). The ```[model_path]``` stands for the huggingface model ID (currently supporting ```huggyllama/llama-7b```, ```mistralai/Mistral-7B-v0.1```, and ```meta-llama/Meta-Llama-3-8B```).
> bash scripts/undefended.sh [model_path] \
> bash scripts/undefended.sh [model_path] run
+ We also support the reproduction of the previous SOTA defense StruQ by defensive SFT
> bash scripts/struq.sh [model_path] \
> bash scripts/struq.sh [model_path] run
+ To tune the learning rate, change lr in LR_CONFIG in ```run.py```

# Test
+ All logs on training, utility evaluation, and security evaluation are saved to ```[model_path]/summary.tsv``` if you use ```bash [script_path] [model_path] run```
+ The default setting is ```--test_attack none ignore completion_real completion_realcmb gcg advp``` (```none``` for utility)
> bash scripts/test.sh [model_path] \
> bash scripts/test.sh [model_path] run
+ Customize the ```--test_attack```, prompting-based ```--defense```, and testing ```--data_path```. The ```--defense``` could be ['none', 'sandwich', 'instructional', 'reminder', 'isolation', 'incontext'], and ```--test_attack``` could be ['naive', 'ignore', 'completion_real', 'completion_realcmb', 'gcg', 'advp']
> python run.py --do_test --test_attack [test_attack1] [test_attack2] [test_attack3] -m [model_path1] [model_path2] [model_path3] -d [defense] --data_path [data_path]
+ This triggers multiple below commands.
> python test.py -a [test_attack] -m [model_path] --defense [defense] --data_path [data_path]
+ Log the GCG and AdvPrompter testing results to ```[model_path]/summary.tsv```. To support this automatic logging, AdvPrompter has to be run through ```bash``` or ```python run.py```, which produces a ```advp_jobID.out``` in ```[model_path]```
> python test.py --log -m [model_path]

# Code Acknowledgements
The majority of SecAlign and the included [StruQ](https://github.com/Sizhe-Chen/StruQ) and [AdvPrompter](https://github.com/facebookresearch/advprompter) are licensed under CC-BY-NC, however portions of the project are available under separate license terms: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) is licensed Apache 2.0; [LLM Attacks](https://github.com/llm-attacks/llm-attacks) is licensed MIT. Code under `gcg/` is adapted from [LLM Attacks](https://github.com/llm-attacks/llm-attacks). Code under `advprompter/` is adapted from [AdvPrompter](https://github.com/facebookresearch/advprompter). This software and/or data was deposited in the BAIR open research Commons repository in 2025.