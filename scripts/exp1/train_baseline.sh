#!/bin/bash

#-------------------------------------
# Configurable Parameters
#-------------------------------------

# wandb settings
wandb_project_prefix="BaselineFP32Smatable"
wandb_mode="online"  # [online, offline, disabled]

# data config
declare -a data_flag_options=("DatabyTable" "DatabyPerson")
declare -a data_splitting_method_list=("PS" "LOSO" "AOS")
declare -a subjects_list=("A" "B" "C")

# model config
declare -a model_type_options=("2dcnn")
p=0.25  # dropout

# experiment settings
num_exps=100
batch_size=10
lr=0.001
num_epochs=100
exp_mode="train"

#-------------------------------------
# Function to run one experiment
#-------------------------------------
run_experiment() {
    local data_flag="$1"
    local model_type="$2"
    local data_split="$3"
    local subject="$4"

    # define save dir
    exp_base_dir="exp_records/fp32/${data_flag}/${model_type}/${data_split}/${subject}"

    # wandb project name (no split method for finetuning)
    wandb_project_name="${wandb_project_prefix}_Waveform"

    echo "▶ Running: data=$data_flag, model=$model_type, split=$data_split, subject=$subject"

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
        --data_splitting_method="$data_split" \
        --target_subject="$subject" 
    }

#-------------------------------------
# Main execution loop
#-------------------------------------
for ((i=1; i<=num_exps; i++)); do
    for data_flag in "${data_flag_options[@]}"; do
        for model_type in "${model_type_options[@]}"; do
            for data_split in "${data_splitting_method_list[@]}"; do
                for subject in "${subjects_list[@]}"; do
                    run_experiment "$data_flag" "$model_type" "$data_split" "$subject"
                done
            done
        done
    done
done
