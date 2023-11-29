import argparse
import os
from pathlib import Path

import torch
import yaml

import wandb
from loop_VAE import train_init_VAE



def train_init(args, device):
    config_path = Path(args.config_path)
    config = yaml.load(config_path.open(mode="r"), Loader=yaml.FullLoader)
    model_to_train = args.model

    if args.use_wandb == True:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config)

        wandbconfig = wandb.config
        wandbconfig.pretrained = args.pretrained
        wandbconfig.use_amp = args.use_amp
        wandbconfig.model = args.model

    if model_to_train == "VAE":
        train_init_VAE(args, config, device)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_init(args, device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", default="train_config.yaml")
    parser.add_argument("--train_normal_dataset_path", default="preprocess/out_preprocess/normaltrain.csv")
    parser.add_argument("--train_anormal_dataset_path", default="preprocess/out_preprocess/anormaltrain.csv")
    parser.add_argument("--model", default="VAE", choices = ["VAE"])
    parser.add_argument('--scheduler', default=True, type=bool)
    parser.add_argument('--pretrained', default=False, type=bool)
    parser.add_argument('--use_amp', default=True, type=bool)
    parser.add_argument('--run_name', default="Test", type=str,
                        help='Name of this run. Used to create folders where to save the weights.')
    parser.add_argument('--use_wandb', default=True, type=bool)
    parser.add_argument('--wandb_project', default='AnomalyDetector_v2', type=str)
    parser.add_argument('--wandb_entity', default='xernpl', type=str)
    parser.add_argument('--show_step', default=100, type=int)
    args = parser.parse_args()

    if not os.path.exists(f'./saved_models_{args.run_name}'):
        os.mkdir(f'./saved_models_{args.run_name}')
    if not os.path.exists(f'./current_models_{args.run_name}'):
        os.mkdir(f'./current_models_{args.run_name}')

    main(args)
