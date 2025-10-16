#!/bin/bash

#-------------------------------------
# Configurable Parameters
#-------------------------------------

# wandb settings
wandb_project_prefix="QuantSmatable"  
wandb_mode="online"  # [online, offline, disabled]

# data config
declare -a data_flag_list=("DatabyPerson" "DatabyTable") 
declare -a data_splitting_method_list=("LOSO") # "AOS" "PS" "LOSO"
declare -a subjects_list=("A" "B" "C")
declare -a downsampling_rate_options=(10) 

# model config
declare -a model_type_options=("1dsepcnnfused") # "1dcnn" "1dsepcnnfused" 
declare -a num_blocks_options=(3) 
p=0  # dropout

# quantization config
declare -a quant_bits_options=(6) # 6 4)


# HW simulation settings
subset_size=1
target_hw="amd" 
fpga_type="xc7s25ftgb196-2" # "xc7s15ftgb196-2" "xc7s25ftgb196-2" "xc7s50ftgb196-2"


# experiment settings
num_exps=50
batch_size=56
lr=0.000330
num_epochs=100
exp_mode="train"

#-------------------------------------
# Function to run one experiment
#-------------------------------------
run_experiment() {
    local data_flag="$1"  
    local model_type="$2"
    local num_blocks="$3"
    local downsampling_rate="$4"
    local data_split="$5"
    local subject="$6"
    local quant_bits="$7"

    # define save dir
    exp_base_dir="exp_records/quant/${data_flag}/${model_type}/${num_blocks}-blocks/${downsampling_rate}-dr/${data_split}/${subject}/${quant_bits}-bits"

    # wandb project name (no split method for finetuning)
    wandb_project_name="${wandb_project_prefix}_Waveform_${data_flag}"

    echo "▶ Running: ${data_flag}, $model_type, blocks=$num_blocks, dr=$downsampling_rate, split=$data_split, subject=$subject, bits=$quant_bits"

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
        --downsampling_rate="$downsampling_rate" \
        --data_splitting_method="$data_split" \
        --target_subject="$subject" \
        --quant_bits="$quant_bits" \
        --subset_size="$subset_size" \
        --target_hw="$target_hw" \
        --fpga_type="$fpga_type" \
        --enable_qat --enable_hw_simulation
    }

#-------------------------------------
# Main execution loop
#-------------------------------------
for ((i=1; i<=num_exps; i++)); do
    for data_flag in "${data_flag_list[@]}"; do
        for model_type in "${model_type_options[@]}"; do
            for downsampling_rate in "${downsampling_rate_options[@]}"; do
                for num_blocks in "${num_blocks_options[@]}"; do
                    for quant_bits in "${quant_bits_options[@]}"; do
                        for data_split in "${data_splitting_method_list[@]}"; do
                            for subject in "${subjects_list[@]}"; do
                                run_experiment "$data_flag" "$model_type" "$num_blocks" "$downsampling_rate" "$data_split" "$subject" "$quant_bits"
                            done
                        done
                    done
                done
            done
        done
    done
done