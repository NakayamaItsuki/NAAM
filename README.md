This is the source code for our "AAAI-2025 Student Abstract and Poster Program" paper: NAAM: Node-Aware Attention Mechanism for Distilling GNNs-to-MLP.

**Note**: This code is based on the code from MTAAM.<br>
*Yang, B. W., Chang, M. Y., Lu, C. H., & Shen, C. Y. 2024. Two Heads Are Better Than One: Teaching MLPs with Multiple Graph Neural Networks via Knowledge Distillation. Database Systems for Advanced Applications (DASFAA).*


## Create environment
```sh
# create env from file
conda env create -f=environment.yml
# activate env
conda activate pyg
```

## Train student models
```sh
# device=0, inductive ratio=0.5
./experiments/NAAM.sh 0 0.5
./experiments/MTAAM.sh 0 0.5
./experiments/GLNN_GCN.sh 0 0.5 # GCN, GAT, SGC, APPNP, GCN2
./experiments/vanilla_MLP.sh 0 0.5 
```

## Show experimental results
```sh
# inductive ratio=0.5
python experiments/results/output_results.py --ratio 0.5
```

## Train teacher models
If you want to train teacher models, use the following code.
```sh
# device=0, inductive ratio=0.5
./run_train_teacher.sh 0 0.5
```
