# Enhancing AI Safety Through the Fusion of Low Rank Adapters



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for "Enhancing AI Safety Through the Fusion of Low Rank Adapters"




## Table of Contents

- [File Structure](#filestructure)
- [Datasets](#datasets)
- [Inside llama2 folder](#insidethellama2folder)
- [Reproducibility](#reproducibility)
- [License](#license)

## FileStructure

    ├── Evaluation Scripts
    │   ├── ....          
    │   ├── ....       
    │   └── ....
    ├── Llama2                  
    │   ├── ckpts         
    │   ├── configs         
    │   ├── finetuned_models
    │   ├── ft_datasets          
    │   ├── inference         
    │   ├── model_checkpointing                
    │   ├── policies        
    │   ├── safety_evaluation
    │   ├── utility_evaluation        
    │   ├── utils       
    │   └── model_checkpointing    
    ├──  llama2_finetuning.ipynb
    ├──  llama2_ft_response_generation_&_evaluation.ipynb
    ├──  llama2_inference.ipynb 
    └──  llama2_merging_adapter.ipynb
    


## Datasets

For the evaluation and the benchmarking of the model, the following datasets have been used-

### 1.HexPhi Dataset

The HexPhi dataset is a specialized safety evaluation benchmark that comprehensively covers 11 harmful categories.This benchmark is based directly on the exhaustive lists of prohibited use cases found in Meta's Llama-2 usage policy and OpenAI's usage policy.We have used this benchmark dataset to evaluate the safety of models.
Please refer to the following github repo on instructions to get access to the dataset-

[LLMs-Finetuning Safety](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety)

###  2.MMLU Dataset

The Massive Multitask Language Understanding (MMLU) dataset is designed to evaluate the multitask learning capabilities of models across a wide range of tasks and domains. MMLU includes tasks that span various fields, including mathematics, science, history, and more.

For the dataset,you could access it from [Here](https://huggingface.co/datasets/cais/mmlu).

The script for running the MMLU dataset is provided in the `Evaluation Scripts` folder


## InsidetheLlama2Folder

The contents of the folder are-
    
    
    ├── Llama2                  
    │   ├── ckpts         
    │   ├── configs         
    │   ├── finetuned_models
    │   ├── ft_datasets          
    │   ├── inference         
    │   ├── model_checkpointing                
    │   ├── policies        
    │   ├── safety_evaluation
    │   ├── utility_evaluation        
    │   ├── utils       
    │   └── model_checkpointing    
    └── ....

### 1.ckpts
The checkpoints that will be generated during finetuning will be stored in this file.

### 2.configs
This folder takes care of the intialization of fsdp, peft, datasets.

### 3.finetuned_models
This folder is where the adapters get stored after finetuning.

### 4.ft_datasets

This folder consists of the following datasets-
  
    1.alpaca

    2.aoa- Consists of the AOA styled controversial prompts

    3.aoa_safety- Consists of the AOA styled safe prompts

    4.dolly
    
    5.pure_bad

    6.HPFA Dataset

    7.ChatAGI Dataset

    8.X_Sum Dataset

Here in case you want to train the Task Adapter(aoa-7b-full) to be either one of HPFA, ChatAGI or X_Sum ,Then just replace the respective `train.json` with the `train.json from aoa dataset folder`

### 5.inference

This folder is designed to convert a model checkpoint saved in Fully Sharded Data Parallel (FSDP) format into a Hugging Face (HF) format. The conversion process allows you to load and save a model in a format that can be easily used with Hugging Face's tools.

### 6.model_checkpointing

This folder handles saving and loading checkpoints for a distributed, sharded model and optimizer state in PyTorch, using FSDP (Fully Sharded Data Parallel) for efficient distributed training.

### 7.policies

This folder applies activation checkpointing to a model using FSDP (Fully Sharded Data Parallel) to optimize memory usage while also allowing flexible precision control for various components 
hese policies configure how model parameters, gradients, and buffers are handled in terms of precision (e.g., float16, bfloat16, or float32)

### 8.safety_evaluation

This folder contains the code for setting up the gpt4 judge for the evaluation of the generated responses.

### 9.utility_evaluation

This folder contains the code for checking the utility of the model for the specific task it has been finetuned on based on the evaluation of resonses on the gpt4 judge.
 
### 10.utils

This folder contains the code for updating and generating configurations for various training and dataset settings, using specified parameters to configure PEFT (Parameter-Efficient Fine-Tuning) and dataset preprocessing options, and provides warnings for unrecognized parameters.

### 11.model_checkpointing

The folder contains the code for storing the model checkpointing during finetuning process.


## Reproducibility

To reproduce our experiments or tinker around, there,there are 4 Jupyter notebooks in this repo, each with a different purpose-

1.[Finetuning your own Adapters](llama2_finetuning.ipynb)

2.[Merging the Adapters with different weight combinations](llama2_merging_adapter.ipynb)

3.[Inference](llama2_inference.ipynb)

4.[Response Generation and Evaluation of model](llama2_ft_response_generation_&_evaluation.ipynb)


## License
`Enhancing AI Safety Through the Fusion of Low Rank Adapters` is licensed under the terms of the MIT license. See LICENSE for more details.
