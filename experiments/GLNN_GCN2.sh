# !bin/bash

device=$1
ratio=$2

agg=attn

base_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
results_dir="${base_dir}/results/${ratio}"
file_path="${results_dir}/GLNN_GCN2.txt"
conf_name="${base_dir}/conf_ratio=${ratio}/GLNN_GCN2_conf.yaml"


mkdir -p "$results_dir"

if [ ! -f "$file_path" ]; then
    touch "$file_path"
fi


### cora ###
python -W ignore train_student.py --agg $agg --dataset cora --hidden_channels 64 \
    --teacher_models_name "GCN2" --teacher_hidden_channels 64 --teacher_layers 64 \
    --ratio $ratio --mode ind \
    --config_name $conf_name \
    --output_only_results True \
    --device $device --n_runs 10 >> $file_path
echo "" >> $file_path


### citeseer ###
python -W ignore train_student.py --agg $agg --dataset citeseer --hidden_channels 64 \
    --teacher_models_name "GCN2" --teacher_hidden_channels 256 --teacher_layers 32 \
    --ratio $ratio --mode ind \
    --config_name $conf_name \
    --output_only_results True \
    --device $device --n_runs 10 >> $file_path
echo "" >> $file_path


### pubmed ###
python -W ignore train_student.py --agg $agg --dataset pubmed --hidden_channels 64 \
    --teacher_models_name "GCN2" --teacher_hidden_channels 256 --teacher_layers 16 \
    --ratio $ratio --mode ind \
    --config_name $conf_name \
    --output_only_results True \
    --device $device --n_runs 10 >> $file_path
echo "" >> $file_path


### amazon_computers ###
python -W ignore train_student.py --agg $agg --dataset amazon_computers --hidden_channels 64 \
    --teacher_models_name "GCN2" --teacher_hidden_channels 256 --teacher_layers 16 \
    --ratio $ratio --mode ind \
    --config_name $conf_name \
    --output_only_results True \
    --device $device --n_runs 10 >> $file_path
echo "" >> $file_path


### amazon_photo ###
python -W ignore train_student.py --agg $agg --dataset amazon_photo --hidden_channels 64 \
    --teacher_models_name "GCN2" --teacher_hidden_channels 256 --teacher_layers 16 \
    --ratio $ratio --mode ind \
    --config_name $conf_name \
    --output_only_results True \
    --device $device --n_runs 10 >> $file_path
echo "" >> $file_path


