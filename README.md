# Learning Composable Chains-of-Thought

This paper provides code for the paper [Learning Composable Chains-of-Thought](https://arxiv.org/abs/2505.22635). 

## Abstract
> A common approach for teaching large language models (LLMs) to reason is to train on chain-of-thought (CoT) traces of in-distribution reasoning problems, but such annotated data is costly to obtain for every problem of interest. We want reasoning models to generalize beyond their training distribution, and ideally to generalize compositionally: combine atomic reasoning skills to solve harder, unseen reasoning tasks. We take a step towards compositional generalization of reasoning skills when addressing a target compositional task that has no labeled CoT data. We find that simply training models on CoT data of atomic tasks leads to limited generalization, but minimally modifying CoT formats of constituent atomic tasks to be composable can lead to improvements. We can train "atomic CoT" models on the atomic tasks with Composable CoT data and combine them with multitask learning or model merging for better zero-shot performance on the target compositional task. Such a combined model can be further bootstrapped on a small amount of compositional data using rejection sampling fine-tuning (RFT). Results on string operations and natural language skill compositions show that training LLMs on Composable CoT outperforms multitask learning and continued fine-tuning baselines within a given training data budget.

## Table of Contents
1. [Installation](#installation)
2. [Data](#data)
3. [Train and Evaluate](#train-and-evaluate)
4. [How to Cite](#how-to-cite)

## Installation
We have tested using Python 3.11.14. Before building the environment, please install the appropriate PyTorch version that corresponds to the hardware configurations (especially GPUs) of your machine here: https://pytorch.org/get-started/locally/
We have tested on a single NVIDIA GH200 GPU with 120G memory using PyTorch 2.5.1, but the training should work on a single GPU of smaller memory as well.

Then, run the following.
```
# Setup virtual environmnet
conda create -n ccot python=3.11
conda activate ccot

# Install LLaMA-Factory
cd ./LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation

# Create the directory to store the finetuned checkpoints and evaluation outputs
mkdir model_ckpt
mkdir outputs
```
## Data
We use string manipulation and arithmetic tasks, and Skill-Mix to evaluate our method. We include the composable CoT datasets we used for each task under ```./data``` as the following:
* ```./data/atomic```: Composable CoT training data for each atomic task.
* ```./data/composition/composable_cot```: Composable CoT training data for two atomic tasks; used for reproducing ComposableCoT-MTL (multitask learning) results in the paper. 
* ```./data/composition/answer_only```: Compositional data with only the answer; used for evaluation.

The data is formatted to be compatible with LLaMA-Factory.

## Train and Evaluate
### Training
We use LLaMA-Factory to finetune Qwen2.5-7B models with Composable CoT data. The training configurations can be found in: ```./scripts/llamafactory```

Training configurations can be run with the following commands using LLaMA-Factory:
```
cd ./LLaMA-Factory
llamafactory-cli train ../scripts/llamafactory/letter_concat_next_last_letter_composable_cot.yaml
```
### Evaluation
After running the training scripts, the evaluation can be run with the following command:
```
bash ./scripts/inference/0shot_composition.sh
```

## How to Cite
If you have any question regarding the code and our work, please feel free to reach out to Fangcong Yin (fangcongyin@utexas.edu).

If you find our work useful, please consider citing us with the following format:
```
@article{Yin2025LearningCC,
  title={Learning Composable Chains-of-Thought},
  author={Fangcong Yin and Zeyu Leo Liu and Liu Leqi and Xi Ye and Greg Durrett},
  journal={ArXiv},
  year={2025},
  volume={abs/2505.22635},
  url={https://api.semanticscholar.org/CorpusID:278960265}
}
```