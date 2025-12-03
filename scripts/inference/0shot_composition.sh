
task="letter_concat_ascii_multiply_answer_only_test";
ntest=700;
model=qwen2.5;
output_dir="./outputs/${model}_${task}_0shot_composition";

CUDA_VISIBLE_DEVICES=0 python inference/inference.py \
    --base_model_name "Qwen/Qwen2.5-7B" \
    --task $task\
    --eval_batch 16 \
    --test_size $ntest \
    --output_dir $output_dir \
    --init_ckpt_name "model_ckpt/letter_concat_ascii_multiply_composable_cot" \
    --insert_prefix_for_inference \
    --insert_suffix_for_inference;


task="letter_concat_next_last_letter_answer_only_test";
ntest=700;
model=qwen2.5;
output_dir="./outputs/${model}_${task}_0shot_composition";

CUDA_VISIBLE_DEVICES=0 python inference/inference.py \
    --base_model_name "Qwen/Qwen2.5-7B" \
    --task $task\
    --eval_batch 16 \
    --test_size $ntest \
    --output_dir $output_dir \
    --init_ckpt_name "model_ckpt/letter_concat_next_last_letter_composable_cot" \
    --insert_prefix_for_inference \
    --insert_suffix_for_inference;


task="next_last_letter_ascii_multiply_answer_only_test";
ntest=700;
model=qwen2.5;
output_dir="./outputs/${model}_${task}_0shot_composition";

CUDA_VISIBLE_DEVICES=0 python inference/inference.py \
    --base_model_name "Qwen/Qwen2.5-7B" \
    --task $task\
    --eval_batch 16 \
    --test_size $ntest \
    --output_dir $output_dir \
    --init_ckpt_name "model_ckpt/next_last_letter_ascii_multiply_composable_cot" \
    --insert_prefix_for_inference \
    --insert_suffix_for_inference;

task="skillmix_literary_rhetorical_answer_only_test";
ntest=245;
model=qwen2.5;
output_dir="./outputs/${model}_${task}_0shot_composition";

CUDA_VISIBLE_DEVICES=0 python inference/inference.py \
    --base_model_name "Qwen/Qwen2.5-7B" \
    --task $task\
    --eval_batch 16 \
    --test_size $ntest \
    --output_dir $output_dir \
    --init_ckpt_name "model_ckpt/skillmix_literary_rhetorical_composable_cot" \
    --insert_prefix_for_inference \
    --insert_suffix_for_inference;