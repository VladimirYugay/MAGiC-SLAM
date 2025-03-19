import argparse

from src.entities.magic_slam import MAGiCSLAM
from src.utils.io_utils import load_config
from src.utils.utils import setup_seed


def get_args():
    parser = argparse.ArgumentParser(description='Arguments to SLAM pipeline')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--input_path', default="")
    parser.add_argument('--output_path', default="")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--group_name', type=str)
    return parser.parse_args()


def update_config_with_args(config, args):
    if args.input_path:
        config["data"]["input_path"] = args.input_path
    if args.output_path:
        config["data"]["output_path"] = args.output_path
    if args.seed:
        config["seed"] = args.seed
    if args.multi_gpu:
        config["multi_gpu"] = True
    if args.use_wandb:
        config["use_wandb"] = True
    if args.wandb_entity:
        config["wandb_entity"] = args.wandb_entity
    if args.wandb_offline:
        config["wandb_offline"] = True
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    if args.group_name:
        config["group_name"] = args.group_name
    return config


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_path)
    config = update_config_with_args(config, args)

    setup_seed(config["seed"])
    coga_slam = MAGiCSLAM(config)
    coga_slam.run()
