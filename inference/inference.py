import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,set_seed
import json
from datasets import Dataset
import pandas as pd
from utils import merge_lora_models,composable_cot_inference,eval_string_operations
import random
import os
import numpy as np
from peft import PeftModel

TASK_INFO_MAP_DIR = './LLaMA-Factory/data/'
TASK_INFO_MAP = json.load(open(os.path.join(TASK_INFO_MAP_DIR,'dataset_info.json')))
## Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--eval_batch',type=int,default=8)
parser.add_argument('--task', type=str,help='The task dataset to train on')
parser.add_argument('--output_dir',type=str,help='The directory of the output file')
parser.add_argument('--init_ckpt_name',type=str,default=None)
parser.add_argument('--ckpt_merge_strategy',type=str,default="linear")
parser.add_argument('--apply_tall_mask', action='store_true')
parser.add_argument('--ties_density',type=float,default=0.2)
parser.add_argument('--merging_weights',type=str,default="1.0")
parser.add_argument('--do_sample',action='store_true',help='Whether to use sampling during training')
parser.add_argument('--temperature',type=float,default=0.9,help='Temperature for sampling')
parser.add_argument('--insert_prefix_for_inference',action='store_true',help='Whether to insert a prefix for inference')
parser.add_argument('--insert_suffix_for_inference',action='store_true',help='Whether to insert a suffix for inference')
parser.add_argument('--num_return_sequences',type=int,default=1,help='Number of samples to generate per input during inference')
parser.add_argument('--base_model_name',type=str,default='llama2-7b-chat',help='The model base to train on',required=True)
parser.add_argument('--torch_dtype',type=str,default='bf16')
parser.add_argument('--device',type=str,default='cuda')
parser.add_argument('--test_size',type=int,default=0)
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--append_label_to_prompt',action='store_true')


args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
set_seed(seed)

eval_batch_size = int(args.eval_batch)
device = args.device
if args.init_ckpt_name is not None:
    init_ckpt_name = args.init_ckpt_name.split(',')

model_name = args.base_model_name
task = args.task 

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if args.torch_dtype == "bf16":
    torch_dtype = torch.bfloat16
else:
    raise("Not implemented yet")
model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map=device, 
                                                torch_dtype=torch_dtype)
merging_weights = args.merging_weights.split(',')

lora_ckpt_dirs = init_ckpt_name

assert task in TASK_INFO_MAP, f"{task} in not supported!"
task_file_name = TASK_INFO_MAP[task]['file_name']
task_file_name = os.path.join(TASK_INFO_MAP_DIR,task_file_name)
task_file = json.load(open(task_file_name))
test_size = args.test_size
if test_size == 0:
    test_size = len(task_file)
if '_test' in task:
    task_file = task_file[:min(test_size,len(task_file))]
elif 'train' in task:
    task_file = random.sample(task_file,min(test_size,len(task_file)))
else:
    raise("Not implemented yet!")
task_df = pd.DataFrame(task_file)
task_dataset = Dataset.from_pandas(task_df)

merged_model = merge_lora_models(
    model = model,
    init_ckpt_name = init_ckpt_name,
    lora_ckpt_dirs = lora_ckpt_dirs,
    ckpt_merge_strategy = args.ckpt_merge_strategy,
    merging_weights = merging_weights,
    ties_density = args.ties_density,
)

max_new_tokens = 256
if "literary" in task or "rhetorical" in task:
    if "_cot" in task:
        max_new_tokens = 1024
    else:
        max_new_tokens = 512
outputs = composable_cot_inference(model=model,
    tokenizer=tokenizer,
    eval_dataset=task_dataset,
    output_dir=args.output_dir,
    num_return_sequences=args.num_return_sequences,
    batch_size=eval_batch_size,
    insert_prefix_for_inference=args.insert_prefix_for_inference,
    insert_suffix_for_inference=args.insert_suffix_for_inference,
    do_sample=args.do_sample,
    temperature=args.temperature,
    max_new_tokens=max_new_tokens,
    answer_separator=" answer:",
    input_key = "instruction",
    label_key = "output",
    append_label_to_prompt = args.append_label_to_prompt
)

if "_ascii_multiply" in task:
    postprocess_method = "first_period_and_digits_only"
    skip_next_letter_shortcut = False
elif "_next_last_letter" in task:
    postprocess_method = "first_period"
    skip_next_letter_shortcut = True
else:
    skip_next_letter_shortcut = False
    postprocess_method = ''

eval_string_operations(
    outputs=outputs,
    num_return_sequences=args.num_return_sequences,
    postprocess_method=postprocess_method,
    prediction_key = "pred",
    label_key = "label",
    skip_next_letter_shortcut = skip_next_letter_shortcut
)


