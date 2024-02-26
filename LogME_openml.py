import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import sys
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import yaml
from pathlib import Path
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from lib.datasets import build_dataset
from model.space_engine import train_one_epoch, evaluate, extract_features
from lib.samplers import RASampler
from lib import utils
from lib.config import cfg, update_config_from_file
import timm
from model.pit_space import pit
from model.autoformer_space import Vision_TransformerSuper
from collections import OrderedDict
import glob
import openml
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import scipy.optimize as opt
import random
from LogME import LogME

def find_yaml_files(folder_path):
    yaml_files = []
    for filename in glob.glob(os.path.join(folder_path, '*config_*.yaml')):
        yaml_files.append(filename)
    return yaml_files

def parse_yaml_file(file_path):
    with open(file_path, 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_data

def load_configs(file_path, max_cfgs):
    folder_path = file_path
    yaml_files = find_yaml_files(folder_path)
    cfgs = []
    i = 0
    if yaml_files:
        for yaml_file in yaml_files:
            cfgs.append(parse_yaml_file(yaml_file))
            i = i + 1
            if(len(cfgs) + 1 > max_cfgs):
                return cfgs
    return cfgs

# Load the list from a file
# Load the list from a file or create an empty one if the file doesn't exist
def load_list_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            loaded_list = [line.strip() for line in file.readlines()]
        if not loaded_list:
            return []

        return loaded_list
    except FileNotFoundError:
        print(f"File '{file_path}' not found. Creating a new one.")
        # Create an empty list and save it to the file
        empty_list = []
        save_list_to_file(file_path, empty_list)
        return empty_list

# Save the list to a file
def save_list_to_file(file_path, data_list):
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(str(item) + '\n')

# Step 4: Create custom PyTorch dataset classes for training and validation
class CustomDataset(Dataset):
    def __init__(self, X, y, base_path, transform=None):
        self.X = X
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(y)
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        file_name = self.X[idx]
        image_path = os.path.join(self.base_path, file_name)

        label = torch.tensor(self.y_encoded[idx], dtype=torch.long)
        # Add a check to skip non-image files
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Skip non-image files
            print('error: ', image_path)
            return torch.rand((3, 224, 224)), label

        try:
            # Open the image only for valid image files
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return torch.rand((3, 224, 224)), label

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Resize the image to match the expected input size
        resize_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image = resize_transform(image)

        # For simplicity, this example assumes 'image' is a placeholder for your image data
        # Convert the PIL image to a torch tensor
        #image = transforms.ToTensor()(image)

        # Use the label encoder to get the encoded label


        return image, label


def get_args_parser():
    parser = argparse.ArgumentParser('Training and Evaluation Script', add_help=False)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    # config file
    parser.add_argument('--cfg',help='experiment configure file name',required=True,type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14, help='max distance in relative position embedding')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_type', default='AUTOFORMER', type=str,
                        help='Type of space to search')

    parser.add_argument('--mode', type=str, default='retrain', choices=['super', 'retrain'], help='mode of AutoFormer')
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')


    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='./data/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10', 'CIFAR100', 'IMNET'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.add_argument('--archs_dir', default='',
                        help='path where architecture configurations are saved')
    parser.set_defaults(amp=True)


    return parser

def main(args):

    utils.init_distributed_mode(args)
    update_config_from_file(args.cfg)
    args.distributed = False



    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    #dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    #dataset_val, _ = build_dataset(is_train=False, args=args)

    # OpenML dataset Large Animals
    #dataset_id = 44320
    #dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)
    #args.data_set = dataset.name
    #base_path = r'C:\Users\vtsim\.openml\org\openml\www\datasets\44320\BRD_Extended\images'

    # OpenML dataset Plants
    #dataset_id = 44318
    #dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)
    #args.data_set = dataset.name
    #base_path = r'C:\Users\vtsim\.openml\org\openml\www\datasets\44318\FLW_Extended\images'


    # OpenML dataset Vehicles
    #dataset_id = 	44329
    #dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)
    #args.data_set = dataset.name
    #base_path = r'C:\Users\vtsim\.openml\org\openml\www\datasets\44329\APL_Extended\images'


    # OpenML dataset OCR
    #dataset_id = 44243
    #dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)
    #args.data_set = dataset.name
    #base_path = r'C:\Users\vtsim\.openml\org\openml\www\datasets\44243\MD_MIX_Micro\images'

    # OpenML dataset Human Actions
    dataset_id = 44319
    dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)
    args.data_set = dataset.name
    base_path = r'C:\Users\vtsim\.openml\org\openml\www\datasets\44319\SPT_Extended\images'



    # Print the dataset name
    print(f"Dataset Name: {args.data_set}")

    df, *_ = dataset.get_data()
    X = df['FILE_NAME'].values
    y = df['CATEGORY'].values
    args.nb_classes = len(set(y))

    print(openml.config.get_cache_directory())



    # Step 3: Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



    # Create instances of the custom datasets for training and validation
    dataset_train = CustomDataset(X_train, y_train, base_path)
    dataset_val = CustomDataset(X_val, y_val, base_path)

    # Step 5: Create data loaders for training and validation
    args.batch_size = 64
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(2 * args.batch_size), num_workers=args.num_workers, pin_memory=args.pin_mem,
        drop_last=False
    )


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    max_cfgs = 10
    cfgs = load_configs(args.archs_dir, max_cfgs)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    for model_id, cfg in enumerate(cfgs):
        if args.model_type == 'AUTOFORMER':
            model_type = args.model_type
            model = Vision_TransformerSuper(img_size=args.input_size,
                                            patch_size=args.patch_size,
                                            embed_dim=cfg['SUPERNET']['EMBED_DIM'], depth=cfg['SUPERNET']['DEPTH'],
                                            num_heads=cfg['SUPERNET']['NUM_HEADS'],mlp_ratio=cfg['SUPERNET']['MLP_RATIO'],
                                            qkv_bias=True, drop_rate=args.drop,
                                            drop_path_rate=args.drop_path,
                                            gp=args.gp,
                                            num_classes=args.nb_classes,
                                            max_relative_position=args.max_relative_position,
                                            relative_position=args.relative_position,
                                            change_qkv=args.change_qkv, abs_pos=not args.no_abs_pos)


            choices = {'num_heads': cfg['SEARCH_SPACE']['NUM_HEADS'], 'mlp_ratio': cfg['SEARCH_SPACE']['MLP_RATIO'],
                       'embed_dim': cfg['SEARCH_SPACE']['EMBED_DIM'], 'depth': cfg['SEARCH_SPACE']['DEPTH']}

            weights_path = "OUTPUT/search/Meta_Album_APL_Extended/model_0_weights.pth"
            state_dict = torch.load(weights_path)
            state_dict.pop('head.weight', None)
            state_dict.pop('head.bias', None)
            #model.load_state_dict(state_dict, strict=False)

        model.to(device)
        teacher_model = None
        teacher_loss = None

        model_ema = None

        model_without_ddp = model
        if args.distributed:

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, model_without_ddp)
        loss_scaler = NativeScaler()
        lr_scheduler, _ = create_scheduler(args, optimizer)

        if args.mixup > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        output_dir = Path(args.output_dir)

        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        # save config for later experiments
        with open(output_dir / "config.yaml", 'w') as f:
            f.write(args_text)
        def copyStateDict(state_dict):
            if list(state_dict.keys())[0].startswith('module'):
                start_idx = 1
            else:
                start_idx = 0
            new_state_dict = OrderedDict()
            for k,v in state_dict.items():
                name = ','.join(k.split('.')[start_idx:])
                new_state_dict[name] = v
            return new_state_dict

        if args.mode == 'retrain' and "RETRAIN" in cfg and model_type == 'AUTOFORMER':
            retrain_config = None
            retrain_config = {'layer_num': cfg['RETRAIN']['DEPTH'], 'embed_dim': [cfg['RETRAIN']['EMBED_DIM']]*cfg['RETRAIN']['DEPTH'],
                              'num_heads': cfg['RETRAIN']['NUM_HEADS'],'mlp_ratio': cfg['RETRAIN']['MLP_RATIO']}
        if args.eval:
            test_stats = evaluate(data_loader_val, model_type, model, device,  mode = args.mode, retrain_config=retrain_config)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            return

        print("Start training")
        start_time = time.time()
        max_accuracy = 0.0

        epochs = args.epochs

        for epoch in range(args.start_epoch, epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            '''train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, model_type, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                amp=args.amp, teacher_model=teacher_model,
                teach_loss=teacher_loss,
                choices=choices, mode = args.mode, retrain_config=retrain_config,
            )'''


            lr_scheduler.step(epoch)
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        # 'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

            #test_stats = evaluate(data_loader_val, model_type, model, device, amp=args.amp, choices=choices, mode = args.mode, retrain_config=retrain_config)
            features,targets = extract_features(data_loader_val, model_type, model, device, amp=args.amp, choices=choices,
                                  mode=args.mode, retrain_config=retrain_config)
            print(f"Features size {np.shape(features)}")
            print(f"Target size {np.shape(targets)}")
            logme = LogME(regression=False)
            # f has shape of [N, D], y has shape [N]
            score = logme.fit(features, targets)
            print(f"LogME score {score}")

            #print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

            '''log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}'''

            '''if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")'''

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and Evaluation Script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)