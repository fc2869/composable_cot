import json,os
import pandas as pd
import torch 
import torch.nn as nn
import numpy as np
import random
import copy
import re
from peft import PeftModel

def composable_cot_inference(model,
    tokenizer,
    eval_dataset,
    output_dir,
    num_return_sequences=1,
    batch_size=2,
    insert_prefix_for_inference=True,
    insert_suffix_for_inference=True,
    do_sample=False,
    temperature=0.9,
    max_new_tokens=512,
    answer_separator=" answer:",
    input_key = "prompt",
    label_key = "label",
    append_label_to_prompt = False
):
    model.eval()
    tokenizer.add_eos_token = False
    tokenizer.padding_side = 'left'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_dir = os.path.join(output_dir,'outputs.json')
    results_json = []
    inputs = eval_dataset[input_key]
    all_labels = eval_dataset[label_key]
    if append_label_to_prompt:
        for i in range(len(inputs)):
            curr_label = all_labels[i]
            curr_label = curr_label.replace('</s>','').replace('<|endoftext|>','')
            if insert_prefix_for_inference:
                inputs[i] = f"{inputs[i]}<prefix> {curr_label} </prefix>"
            else:
                inputs[i] = f"{inputs[i]} {curr_label}Explanation:"
    if insert_prefix_for_inference:
        for i in range(len(inputs)):
            inputs[i] = inputs[i] + '<prefix>'
    iterator = range(0, len(inputs), batch_size)
    generated = []
    with torch.no_grad():
        for i in iterator:
            inputs_b = inputs[i:i+batch_size]
            inputs_b = tokenizer(inputs_b, return_tensors='pt',padding=True)
            inputs_b= {k:v.to(model.device) for (k,v) in inputs_b.items()}

            if not do_sample:
                outputs = model.generate(**inputs_b,max_new_tokens=max_new_tokens,do_sample=False)
            else:
                outputs = model.generate(**inputs_b,max_new_tokens=max_new_tokens,do_sample=True,temperature=temperature, num_return_sequences=num_return_sequences)

            
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            generated.extend(decoded_outputs)
    if insert_suffix_for_inference:
        
        prefix_generation = copy.deepcopy(generated)
        generated = []
        iterator = range(0, len(prefix_generation), batch_size)
        with torch.no_grad():
            for i in iterator:
                inputs_b = prefix_generation[i:i+batch_size]
                inputs_b = [ip + ' <suffix>' for ip in inputs_b]

                inputs_b = tokenizer(inputs_b, return_tensors='pt',padding=True)
                inputs_b= {k:v.to(model.device) for (k,v) in inputs_b.items()}
                if not do_sample:
                    outputs = model.generate(**inputs_b,max_new_tokens=max_new_tokens,do_sample=False)
                else:
                    ## We do not resample for multiple generations for each suffix, so num_return_sequences is set to 1
                    outputs = model.generate(**inputs_b,max_new_tokens=max_new_tokens,do_sample=True,temperature=temperature, num_return_sequences=1)
                
                decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generated.extend(decoded_outputs)
    
    corr=0
    for i in range(len(generated)):
        
        seq = generated[i]
        sep = answer_separator
        ans = seq.split(sep)[-1].lstrip().rstrip().lower()
        entry_corr = 0
        label = eval_dataset[label_key][i//num_return_sequences]
        if ans == label.lower():
            entry_corr=1
        corr+=entry_corr
        

        result = {'prompt': inputs[i//num_return_sequences],'response':seq,'pred':ans,'label':label}
        results_json.append(result)

    json.dump(results_json,open(results_dir,'w'))
    print('Saved results to:',results_dir)
    return results_json

def merge_lora_models(
    model,
    init_ckpt_name: [str],
    lora_ckpt_dirs: [str],
    ckpt_merge_strategy: str = "linear",
    merging_weights: [float] = [1.0],
    ties_density: float = 0.9,
    majority_sign_method: str = "total"


):
    for i,ckpt_name in enumerate(init_ckpt_name):
        peft_path = lora_ckpt_dirs[i]
        print(peft_path)
        if i == 0:
            model = PeftModel.from_pretrained(model,peft_path,adapter_name = ckpt_name)
        else:
            model.load_adapter(peft_path,adapter_name = ckpt_name)
    print('Loading adapters')
    if len(init_ckpt_name) == 1:
        ## If there is only one model, do not perform model merging
        model.set_adapter(init_ckpt_name[0])
    else:
        assert ckpt_merge_strategy in ['ties','linear','ties_svd']

        if len(merging_weights) == 1:
            merging_weights = [float(merging_weights[0])] * len(init_ckpt_name)
        else:
            merging_weights = [float(w) for w in merging_weights]

        new_adapter_name = "merge"
        if ckpt_merge_strategy == 'ties':
            density = ties_density
            model.add_weighted_adapter(init_ckpt_name, merging_weights, new_adapter_name, combination_type=ckpt_merge_strategy, density=density, majority_sign_method=majority_sign_method)
        elif ckpt_merge_strategy == 'linear':
            model.add_weighted_adapter(init_ckpt_name, merging_weights, new_adapter_name,combination_type=ckpt_merge_strategy)
        elif ckpt_merge_strategy == 'ties_svd':
            density = ties_density
            model.add_weighted_adapter(init_ckpt_name, merging_weights, new_adapter_name, combination_type=ckpt_merge_strategy, density=density, majority_sign_method=majority_sign_method)
        model.set_adapter(new_adapter_name)
    model = model.merge_and_unload(progressbar=True)
    return model

def eval_string_operations(
    outputs: [],
    num_return_sequences=1,
    postprocess_method:str=None,
    prediction_key = "pred",
    label_key = "label",
    remove_eos_tokens = True,
    print_parsing_results = False,
    skip_next_letter_shortcut = False
):


    data = outputs
    corr = 0
    n_samples = num_return_sequences
    answer_indicators = ['the answer is', 'gives us']
    total = 0

    if skip_next_letter_shortcut:
        data = [d for d in data if not 'Take the last letter' in d['prompt']]
    
    for i in range(0,len(data)+1,n_samples):
        ex_corr = 0
        for j in range(i, min(i+n_samples, len(data))):
            if ex_corr:
                break
            prompt = data[j]['prompt']
            
            if skip_next_letter_shortcut and 'Take the last letter' in prompt:
                continue
            if n_samples == 1:
                total+=1
            entry = data[j]
            pred = entry[prediction_key]
            if remove_eos_tokens:
                pred = pred.replace('</s>','').replace('<|endoftext|>','')
            label = entry[label_key]
            if remove_eos_tokens:
                label = label.replace('</s>','').replace('<|endoftext|>','')
            label_ans = label
            
            for answer_indicator in answer_indicators:
                predicted_ans = pred.lower().split(answer_indicator)
                
                if len(predicted_ans) > 1:
                    ## Only keep the last part after the answer indicator
                    predicted_ans = predicted_ans[-1].lstrip().rstrip().lstrip().rstrip()
                    
                    if "first_period" in postprocess_method:

                        ## Seperated by '.'
                        predicted_ans = predicted_ans.split('.')[0]
                    if "digits_only" in postprocess_method:
                        ## Only keep the digits
                        predicted_ans = re.sub(r'\D', '', predicted_ans)
                    
            
                
                if predicted_ans == label_ans:
                    corr+=1
                    ex_corr = 1
                    
                    break
    if n_samples > 1:
        total = len(data)
        
    print(f'Answer Accuracy: {corr}/{(total//n_samples)} = {corr/(total//n_samples)}')