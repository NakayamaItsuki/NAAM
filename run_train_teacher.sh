#!/bin/bash 

datasets=("cora" "citeseer" "pubmed" "amazon_photo" "amazon_computers")
log_path="saves/teacher_results"
mkdir -p $log_path
device=$1
ratio=${2:-1.0}

train_teacher() {
    local mode

    # if ratio is 1.0, then mode is tran else ind
    if [ "$ratio" == "1.0" ]; then
        mode="tran"
    else
        mode="ind"
    fi

    python train_teacher.py \
        --dataset=$dataset \
        --model=$model \
        --device=$device \
        --mode=$mode \
        --ratio=$ratio \
         | tee -a "${log_path}/${dataset}_${mode}.txt"
}

for dataset in "${datasets[@]}"
do
    for model in GCN GAT SGC APPNP GCN2
    do
        train_teacher
    done
done
