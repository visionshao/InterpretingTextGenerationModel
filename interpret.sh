data_path=/mnt/ai2lab/weishao4/programs/SINMT_LLM/T5/data/data_preprocess
mkdir -p /mnt/ai2lab/weishao4/programs/SINMT_LLM/T5/results

# ldc zhen mt5_large
nmt_checkpoint=/mnt/ai2lab/weishao4/programs/SINMT_LLM/T5/save_checkpoints/ldc_zhen_mt5_large/checkpoint-40500
sinmt_checkpoint=/mnt/ai2lab/weishao4/programs/SINMT_LLM/T5/save_checkpoints/ldc_zhen_interpreter_mt5_large_src_batt_last_1_to_6/checkpoint-750
output_dir=/mnt/ai2lab/weishao4/programs/SINMT_LLM/T5/results/interpretation
dataset_name=ldc
model_type=mt5_large
src_lang=zh
tgt_lang=en
bs=8 # gradient
bs=16

CUDA_VISIBLE_DEVICES=4 python3 ../src/interpreting.py \
    --data_path $data_path \
    --dataset_name $dataset_name \
    --bs $bs \
    --src_lang $src_lang \
    --tgt_lang $tgt_lang \
    --nmt_checkpoint $nmt_checkpoint \
    --beam 5 \
    --model_type $model_type \
    --output_dir $output_dir


# ldc zhen mt5_large_interperter
# model_type=mt5_large_interpreter
# CUDA_VISIBLE_DEVICES=4 python3 ../src/interpreting.py \
#     --data_path $data_path \
#     --dataset_name $dataset_name \
#     --bs $bs \
#     --src_lang $src_lang \
#     --tgt_lang $tgt_lang \
#     --nmt_checkpoint $nmt_checkpoint \
#     --sinmt_checkpoint $sinmt_checkpoint \
#     --beam 5 \
#     --model_type $model_type \
#     --output_dir $output_dir


# wmt14 ende mt5_large
# nmt_checkpoint=/mnt/ai2lab/weishao4/programs/SINMT_LLM/T5/save_checkpoints/wmt14_ende_mt5_large/checkpoint-90000
# # sinmt_checkpoint=/mnt/ai2lab/weishao4/programs/SINMT_LLM/T5/save_checkpoints/wmt_zhen_interpreter_mt5_large_src_batt_last_1_to_6/checkpoint-750
# output_dir=/mnt/ai2lab/weishao4/programs/SINMT_LLM/T5/results/interpretation
# dataset_name=wmt14
# model_type=mt5_large
# src_lang=en
# tgt_lang=de
# bs=16

# CUDA_VISIBLE_DEVICES=2 python3 ../src/interpreting.py \
#     --data_path $data_path \
#     --dataset_name $dataset_name \
#     --bs $bs \
#     --src_lang $src_lang \
#     --tgt_lang $tgt_lang \
#     --nmt_checkpoint $nmt_checkpoint \
#     --beam 5 \
#     --model_type $model_type \
#     --output_dir $output_dir




# wmt14 deen mt5_large
# nmt_checkpoint=/mnt/ai2lab/weishao4/programs/SINMT_LLM/T5/save_checkpoints/wmt14_deen_mt5_large/checkpoint-30500
# # sinmt_checkpoint=/mnt/ai2lab/weishao4/programs/SINMT_LLM/T5/save_checkpoints/wmt_zhen_interpreter_mt5_large_src_batt_last_1_to_6/checkpoint-750
# output_dir=/mnt/ai2lab/weishao4/programs/SINMT_LLM/T5/results/interpretation
# dataset_name=wmt14
# model_type=mt5_large
# src_lang=de
# tgt_lang=en
# bs=16

# CUDA_VISIBLE_DEVICES=3 python3 ../src/interpreting.py \
#     --data_path $data_path \
#     --dataset_name $dataset_name \
#     --bs $bs \
#     --src_lang $src_lang \
#     --tgt_lang $tgt_lang \
#     --nmt_checkpoint $nmt_checkpoint \
#     --beam 5 \
#     --model_type $model_type \
#     --output_dir $output_dir