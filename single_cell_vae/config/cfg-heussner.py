import ml_collections
import os
import torch
from torchvision.transforms import (
    Compose,
    Resize,
)


def get_config():
    cfg = ml_collections.ConfigDict()

    cfg.deterministic = False
    cfg.max_epochs = 1000
    cfg.batch_size = 4
    cfg.manual_seed = 100
    cfg.device = "cuda"
    cfg.accelerator = "gpu"
    cfg.num_gpus = 1
    cfg.accel_strategy = "ddp" #if cfg.num_gpus >= 2 else None
    cfg.detect_anomaly = False
    cfg.sample_step = 10  # every sample_step batches model logs

    # data
    cfg.datapath = "/var/local/ChangLab/train/"
    cfg.img_size = (128,128)
    cfg.data_params = {
        "dataset": {"transform": Resize(cfg.img_size),},
        "loader": {
            "batch_size": cfg.batch_size,
            "shuffle": False if cfg.deterministic else True,
            "pin_memory": False,
            "num_workers": 0,
        },
    }

    # optimizer
    cfg.optim = ml_collections.ConfigDict()
    cfg.optim.type = "adam"
    cfg.optim.lr = 0.00005
    cfg.optim.weight_decay = 0.0
    cfg.optim.scheduler = None
    cfg.optim.scheduler_params = None

    # logging details
    cfg.logging = ml_collections.ConfigDict()
    cfg.logging.deterministic = True
    cfg.logging.name = "2023_04_13"
    cfg.logging.debug = True
    cfg.logging.save_dir = (
        "/home/groups/ChangLab/heussner/single-cell-vae/single_cell_vae/logs"
    )
    cfg.logging.fix_version = True

    # early stopping
    cfg.early_stopping = ml_collections.ConfigDict()
    cfg.early_stopping.do = True
    cfg.early_stopping_params = {
        "monitor": "loss",
        "mode": "min",
        "min_delta": 0.00001,
        "patience": 20,
        "check_finite": True,
        "verbose": True,
    }

    # auto checkpointing
    cfg.auto_checkpoint = True
    cfg.checkpoint_dir = "checkpoints"
    cfg.checkpoint_monitor = "loss"
    cfg.checkpoint_mode = "min"
    cfg.load_checkpoint = False
    cfg.model_path = None

    cfg.eval_mode = False
    cfg.eval_model_path = "/home/groups/ChangLab/heussner/single-cell-vae/single_cell_vae/logs/2023_04_12/version_0/checkpoints/last.ckpt"
    cfg.results_dir = (
        "/home/groups/ChangLab/heussner/single-cell-vae/single_cell_vae/logs/2023_04_12/version_0/results/"
    )
    cfg.vae_embed_file = "vae.csv"
    cfg.tsne_embed_file = "tsne.csv"
    cfg.umap_embed_file = "umap.csv"

    return cfg
