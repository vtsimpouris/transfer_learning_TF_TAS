#!/bin/bash
python search_autoformer.py --data-path './dataset/CIFAR10' --data-set 'CIFAR10' --gp  --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search' --population-num 8



