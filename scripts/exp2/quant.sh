#!/bin/bash

#-------------------------------------
# Configurable Parameters
#-------------------------------------

# wandb settings
wandb_project_prefix="Opt"
wandb_mode="offline"  # [online, offline, disabled]

# data config
data_flag="DatabyPerson"  # or "DatabyTable"
declare -a data_splitting_method_list=("AOS")  # "PS" "LOSO" "AOS"
declare -a subjects_list=("A")  # "B" "C"
declare -a downsampling_rate_options=(10)

# model config
declare -a model_type_options=("1dsepcnnfused") # "1dcnn" "1dsepcnnfused" 
p=0  # dropout

# HW simulation settings
subset_size=1
target_hw="amd" 
fpga_type="xc7s25ftgb196-2" # "xc7s15ftgb196-2" "xc7s25ftgb196-2" "xc7s50ftgb196-2"

# experiment settings
num_epochs=100
exp_mode="train"

# optuna search settings
n_trials=100
optuna_hw_target="energy" # "energy" "latency"

#-------------------------------------
# Function to run one experiment
#-------------------------------------
run_experiment() {
    local model_type="$1"
    local downsampling_rate="$2"
    local data_splitting_method="$3"
    local subject="$4"

    # define save dir
    exp_base_dir="exp_records_optxxxxx/quant/${data_flag}/${model_type}/${downsampling_rate}-dr/${subject}/${data_splitting_method}"

    # wandb project name (no split method for finetuning)
    wandb_project_name="${wandb_project_prefix}_${data_flag}_${data_splitting_method}_${model_type}"

    echo "▶ Running: data=$data_flag, model=$model_type, dr=$downsampling_rate, split=$data_splitting_method, subject=$subject"

    python optuna_search.py \
        --wandb_project_name="$wandb_project_name" \
        --wandb_mode="$wandb_mode" \
        --data_flag="$data_flag" \
        --exp_mode="$exp_mode" \
        --exp_base_dir="$exp_base_dir" \
        --num_epochs="$num_epochs" \
        --model_type="$model_type" \
        --p="$p" \
        --downsampling_rate="$downsampling_rate" \
        --data_splitting_method="$data_split" \
        --target_subject="$subject" \
        --subset_size="$subset_size" \
        --target_hw="$target_hw" \
        --fpga_type="$fpga_type" \
        --enable_qat \
        --n_trials="$n_trials" \
        --optuna_hw_target="$optuna_hw_target" \
        --enable_hw_simulation
    }

#-------------------------------------
# Main execution loop
#-------------------------------------
for model_type in "${model_type_options[@]}"; do
    for downsampling_rate in "${downsampling_rate_options[@]}"; do
        for data_split in "${data_splitting_method_list[@]}"; do
            for subject in "${subjects_list[@]}"; do
                run_experiment "$model_type" "$downsampling_rate" "$data_split" "$subject"
            done
        done
    done
done
