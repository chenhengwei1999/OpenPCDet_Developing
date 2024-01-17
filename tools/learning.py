from pcdet.config import cfg_from_yaml_file
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from pcdet.config import cfg


cfg_file = 'cfgs/kitti_models/pointpillar.yaml'

cfg_from_yaml_file(cfg_file, cfg)

train_set, train_loader, train_sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=16,
    dist=False,
    workers=4,
    logger=None,
    training=True,
    merge_all_iters_to_one_epoch=False,
    total_epochs=100,
    seed=666)

print("OK")