# Steps to Evaluate the model against BenchMarks


## NOTE-Create a results folder  and use it as the save_dir path

### 1. Evaluation for Finetuned Model

```!python3 /path_to_eval_for_adapter_model.py --ntrain 5 --ngpu 1 --data_dir /path_to_dataset --save_dir /path_to_results_folder --model /path_to_model --quantization --use_cache ```


### Eg:-

```!python3 /content/safety-finetuning-llm/llama2/eval.py --ntrain 5 --ngpu 1 --data_dir /content/drive/MyDrive/MMLU/data --save_dir /content/safety-finetuning-llm/results  --model /content/safety-finetuning-llm/llama2/finetuned_models/aoa-7b-full --quantization --use_cache ```

### 2. Evaluation for Base Model

```!python3 /path_to_eval_for_base_model.py --ntrain 5 --ngpu 1 --data_dir /path_to_dataset --save_dir \path_to_results_folder --model name_of_huggingface_model --quantization --use_cache ```

### Eg:-

```!python3 /content/safety-finetuning-llm/llama2/eval.py --ntrain 5 --ngpu 1 --data_dir /content/drive/MyDrive/MMLU/data --save_dir /content/safety-finetuning-llm/results  --model meta-llama/Llama-2-7b-chat-hf  --quantization --use-cache ```