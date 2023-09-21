#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path './dataset/CIFAR10' --data-set 'CIFAR10' --gp --change_qk --relative_position \
--mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/subnet_autoformer/TF_TAS-T.yaml' --output_dir './OUTPUT/sample'
#python train.py --data-path './dataset/CIFAR10' --data-set 'CIFAR10' --gp --change_qk --relative_position --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg 'OUTPUT/search/data_0.yaml' --output_dir './OUTPUT/sample'
#python train_multiple_archs.py --data-path './dataset/CIFAR10' --data-set 'CIFAR10' --gp --change_qk --relative_position --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg 'OUTPUT/search/data_0.yaml' --archs_dir 'OUTPUT/search' --output_dir './OUTPUT/sample' --epochs 1
