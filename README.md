# MLDT: Multi-Level Decomposition for Complex Long-Horizon Robotic Task Planning with Open-Source Large Language Model

> **Abstract**
In the realm of data-driven AI technology, the application of open-source large language models (LLMs) in robotic task planning represents a significant milestone. Recent robotic task planning methods based on open-source LLMs typically leverage vast task planning datasets to enhance models’ planning abilities. While these methods show promise, they struggle with complex long-horizon tasks, which require comprehending more context and generating longer action sequences. This paper addresses this limitation by proposing MLDT, the Multi-Level Decomposition Task planning method. This method innovatively decomposes tasks at the goal-level, task-level, and action-level to mitigate the challenge of complex long-horizon tasks. In order to enhance open-source LLMs’ planning abilities, we introduce a goal-sensitive corpus generation method to create high-quality training data and conduct instruction tuning on the generated corpus. Since the complexity of the existing datasets is not high enough, we construct a more challenging dataset, LongTasks, to specifically evaluate planning ability on complex long-horizon tasks. We evaluate our method using various LLMs on four datasets in VirtualHome. Our results demonstrate a significant performance enhancement in robotic task planning, showcasing MLDT’s effectiveness in overcoming the limitations of existing methods based on open-source LLMs as well as its practicality in complex, real-world scenarios.
> 
![](./figs/1.png)
This is the accompanying code for the paper "MLDT: Multi-Level Decomposition for Complex Long-Horizon Robotic Task Planning with Open-Source Large Language Model".

## Setup
### Environment Setup
```
conda create -n MLDT python=3.10
conda activate MLDT
pip install -r requirement.txt
```
### VirtualHome Setup
We conduct experiments in VirtualHome. Download VirtualHome exectuable file (v2.2.5) from [here](https://1drv.ms/u/s!Am9fgKqXV2C2bB8WJWKb4-NABSg?e=8FJOUA) and unzip it to ```MLDT/MLDT/virtualhome/```. 
```
MLDT/
└── MLDT/
    └── virtualhome/                                                          
```
### LLM Setup
We employ various LLMs of different scales as the backbone, including Llama-2-chat([7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)), bloom([3B](https://huggingface.co/bigscience/bloom-3b), [7B](https://huggingface.co/bigscience/bloom-7b1)), [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b). To examine the effectiveness of our method on long-context LLMs, we use LongAlpaca ([7B](https://huggingface.co/Yukang/LongAlpaca-7B), [13B](https://huggingface.co/Yukang/LongAlpaca-13B)), [ChatGLM3-6B-32K](https://huggingface.co/THUDM/chatglm3-6b-32k), the long-context
versions of Llama-2-chat and ChatGLM, respectively. Download LLMs to ```MLDT/pretrain/```.
```
MLDT/
└── pretrain/
    ├── bloom-3b/
    ├── bloom-7b/
    ├── chatglm3-6b/
    ├── chatglm3-6b-32k/
    ├── llama-2-7b-chat-hf/
    ├── llama-2-13b-chat-hf/
    ├── LongAlpaca-7B/
    └── LongAlpaca-13B/                         
```

## Dataset
We evaluate our method on four datasets, three (In-Distribution, NovelScenes, NovelTasks) from [LID](https://github.com/ShuangLI59/Pre-Trained-Language-Models-for-Interactive-Decision-Making) and one (LongTasks) created by ourselves. Download the four datasets from [here](https://1drv.ms/u/s!AvfJPiUjTsi_aYQaFwohMS7NA2s?e=tZkalm) and unzip it to ```MLDT/MLDT/data/test_init_env/```
```
MLDT/
└── MLDT/
    ├── data/                  
        └── test_init_env/                                        
```                                                     
You can run ```MLDT/MLDT/data/create_long.py``` to create the LongTasks dataset. We remove some samples due to the environment bug from LID.

## Goal-sensitive Corpus Generation
We provide all the instruction datasets for different methods in ```MLDT/Instruction-Tuning/```: "multi-layer" for "MLDT", "react" for "ReAct", "embodied" for "Embodied", "task-action" for "MLDT<sub>-goal</sub>", "goal-action" for "MLDT<sub>-task</sub>".
You can go to ```MLDT/MLDT/task-planning/``` and run ```bash scripts/llm_collection.sh``` to generate the training corpus for "MLDT". You can modify the parameters like "subset", "max_retry" to generate your own data. For other methods, you can either modify the python scripts to generate training corpus from scratch or use some tricks like regular expressions to obtain the training corpus based on the generated corpus.

## Instruction Tuning
Go to ```MLDT/Instruction-Tuning/```. Run ```run_bloom-3b.sh``` or ```run_bloom-7b.sh``` for fine-tuning bloom. Run ```run_chatglm.sh``` for fine-tuning ChatGLM3-6B or ChatGLM3-6B-32K. Run ```run_llama-7b.sh``` or ```run_llama-13b.sh``` for fine-tuning Llama-2-chat or LongAlpaca. You can modify the parameters like "dataset", "train_batch_size", "accumulation_steps" to fit your own training.

## Multi-Level Decomposition for Robotic Task Planning
### Command
Go to ```MLDT/MLDT/task-planning/```. Run ```bash scripts/llm_eval.sh``` to evaluate open-source LLMs for "MLDT", "Embodied", "ReAct", "MLDT<sub>-goal</sub>", and "MLDT<sub>-task</sub>". Run ```bash scripts/llm_eval_demo.sh``` to evaluate open-source LLMs for "MLDT<sub>-ft</sub>". Run ```bash scripts/gpt_eval.sh``` to evaluate closed-source LLMs for "MLDT", "Embodied", "ReAct".
### Key Parameters
- base_port: port number for VirtualHome environment
- llm: LLM backbone location
- lora: lora weight location, "None" for using LLM backbone only or closed-source LLMs
- mode: select task planning method: "multi-layer" for "MLDT", "react" for "ReAct", "embodied" for "Embodied", "goal-action" for "MLDT<sub>-task</sub>", "task-action" for "MLDT<sub>-goal</sub>"
- api: API key for ChatGPT
- demo: add this to use demonstrations
- max_retry: the number of times that task planning models can try, we set 1 in our experiment, you can set larger for higher sucess rate but longer inference time, it is useful for generating more training corpus
### Lora Checkpoint
