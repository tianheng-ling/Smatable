#!/bin/bash

#-------------------------------------
# Configurable Parameters
#-------------------------------------

# wandb settings
wandb_project_prefix="ParamFP32Smatable"
wandb_mode="disabled"  # [online, offline, disabled]

# data config
data_flag="DatabyPerson"  # or "DatabyTable"
declare -a data_splitting_method_list=("PS") # "LOSO" "AOS" 
declare -a subjects_list=("A") # "B" "C")
declare -a normalization_type_options=("standard")  # "minmax" "maxabs"
declare -a downsampling_rate_options=(10)

# model config
declare -a model_type_options=("1dcnn") # "2dcnn" "1dcnn" "1dsepcnn" 
declare -a num_blocks_options=(3)  # number of blocks in the model
p=0  # dropout

# experiment settings
num_exps=1
batch_size=32
lr=0.001
num_epochs=1
exp_mode="train"

#-------------------------------------
# Function to run one experiment
#-------------------------------------
run_experiment() {
    local model_type="$1"
    local num_blocks="$2"
    local normalization_type="$3"
    local downsampling_rate="$4"
    local data_split="$5"
    local subject="$6"

    # define save dir
    exp_base_dir="exp_records_test/fp32/${data_flag}/${model_type}/${num_blocks}-blocks/${downsampling_rate}-dr/${data_split}/${subject}"

    # wandb project name (no split method for finetuning)
    wandb_project_name="${wandb_project_prefix}_Waveform_${data_flag}"

    echo "▶ Running: $model_type, blocks=$num_blocks, subject=$subject, norm=$normalization_type, dr=$downsampling_rate, split=$data_split"

    python main.py \
        --wandb_project_name="$wandb_project_name" \
        --wandb_mode="$wandb_mode" \
        --data_flag="$data_flag" \
        --batch_size="$batch_size" \
        --lr="$lr" \
        --exp_mode="$exp_mode" \
        --exp_base_dir="$exp_base_dir" \
        --num_epochs="$num_epochs" \
        --model_type="$model_type" \
        --p="$p" \
        --num_blocks="$num_blocks" \
        --normalization_type="$normalization_type" \
        --downsampling_rate="$downsampling_rate" \
        --data_splitting_method="$data_split" \
        --target_subject="$subject" 
    }

#-------------------------------------
# Main execution loop
#-------------------------------------
for ((i=1; i<=num_exps; i++)); do
    for model_type in "${model_type_options[@]}"; do
        for num_blocks in "${num_blocks_options[@]}"; do
            for normalization_type in "${normalization_type_options[@]}"; do
                for downsampling_rate in "${downsampling_rate_options[@]}"; do
                    for data_split in "${data_splitting_method_list[@]}"; do
                        for subject in "${subjects_list[@]}"; do
                            run_experiment "$model_type" "$num_blocks" "$normalization_type" "$downsampling_rate" "$data_split" "$subject"
                        done
                    done
                done
            done
        done
    done
done
