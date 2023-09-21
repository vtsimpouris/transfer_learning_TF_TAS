#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env search_autoformer.py --data-path './dataset/CIFAR10' --data-set 'CIFAR10' --gp \
 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search'
while true
do
    # Your script's logic here
    sleep 1  # Optional delay to avoid high CPU usage
done
#python search_autoformer.py --data-path './dataset/CIFAR10' --data-set 'CIFAR10' --gp  --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search' --population-num 8



