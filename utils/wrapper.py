import os
import math
import torch

from train import SitsScdWrapper
from data.datamodule import ImageDataModule
from data.data import DynamicEarthNet, Muds, DynamicEarthNetMSI
import yaml
from models.losses import Losses
from models.st3dunet import ST3DUNet
from metrics.scd_metrics import SCDMetric
from utils.delete import clean_empty_ckpt_dirs, clean_empty_visual_dirs, clean_orphan_log_dirs
from visualize import Tester


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_model(cfg):

    config = cfg['ST3DUNET']
    config_spatial, config_cross = config['SPATIAL_ATTENTION'], config['CROSS_ATTENTION']
    model = ST3DUNet(
        input_dim=config['INPUT_CHANNELS'],
        output_dim=config['OUTPUT_CHANNELS'],
        encoder_widths=config['ENCODER_WIDTHS'],
        decoder_widths=config['DECODER_WIDTHS'],
        str_conv_k=config['STR_CONV_K'],
        str_conv_s=config['STR_CONV_S'],
        str_conv_p=config['STR_CONV_P'],
        n_head=config['N_HEAD'],
        d_model=config['D_MODEL'],
        d_k=config['D_K'],
        mlp=config['MLP'],

        image_size=config_spatial['IMAGE_SIZE'],
        patch_size_spatial=config_spatial['PATCH_SIZE'],
        depth_spatial=config_spatial['DEPTH'],
        heads_spatial=config_spatial['N_HEAD'],
        emb_dim_spatial=config_spatial['EMB_DIM'],
        dropout=config_spatial['DROPOUT'],

        heads_cross=config_cross['N_HEAD'],
        dropout_cross=config_cross['DROPOUT'],
    )

    return model

def load_loader(cfg):
    if cfg['dataset']['name'].lower() == 'muds':
        trainset = Muds(
            path=cfg['dataset']['path'],
            split="train",
            num_channels=cfg['dataset']['num_channels'],
            num_classes=cfg['dataset']['num_classes'],
            img_size=cfg['dataset']['image_size'],
            true_size=cfg['dataset']['true_size'],
            train_length=cfg['dataset']['seq_len'],
            sampling_mode=cfg['dataset']['sampling_mode'],
        )
        testset_in = Muds(
            path=cfg['dataset']['path'],
            split="val",
            num_channels=cfg['dataset']['num_channels'],
            num_classes=cfg['dataset']['num_classes'],
            img_size=cfg['dataset']['image_size'],
            true_size=cfg['dataset']['true_size'],
            train_length=cfg['dataset']['seq_len'],
            date_aug_range=cfg['dataset']['data_aug_range'],
            sampling_mode=cfg['dataset']['sampling_mode'],
            domain_shift=False,
        )
        testset_out = Muds(
            path=cfg['dataset']['path'],
            split="val",
            num_channels=cfg['dataset']['num_channels'],
            num_classes=cfg['dataset']['num_classes'],
            img_size=cfg['dataset']['image_size'],
            true_size=cfg['dataset']['true_size'],
            train_length=cfg['dataset']['seq_len'],
            date_aug_range=cfg['dataset']['data_aug_range'],
            sampling_mode=cfg['dataset']['sampling_mode'],
            domain_shift=True,
        )

    elif cfg['dataset']['name'].lower() == 'dynamicearthnet':
        trainset = DynamicEarthNet(
            path=cfg['dataset']['path'],
            split="train",
            num_channels=cfg['dataset']['num_channels'],
            num_classes=cfg['dataset']['num_classes'],
            img_size=cfg['dataset']['image_size'],
            true_size=cfg['dataset']['true_size'],
            train_length=cfg['dataset']['seq_len'],
            date_aug_range=cfg['dataset']['data_aug_range'],
            sampling_mode=cfg['dataset']['sampling_mode'],
        )
        testset_in = DynamicEarthNet(
            path=cfg['dataset']['path'],
            split="val",
            num_channels=cfg['dataset']['num_channels'],
            num_classes=cfg['dataset']['num_classes'],
            img_size=cfg['dataset']['image_size'],
            true_size=cfg['dataset']['true_size'],
            train_length=cfg['dataset']['seq_len'],
            date_aug_range=cfg['dataset']['data_aug_range'],
            sampling_mode=cfg['dataset']['sampling_mode'],
            domain_shift=False,
        )
        testset_out = DynamicEarthNet(
            path=cfg['dataset']['path'],
            split="val",
            num_channels=cfg['dataset']['num_channels'],
            num_classes=cfg['dataset']['num_classes'],
            img_size=cfg['dataset']['image_size'],
            true_size=cfg['dataset']['true_size'],
            train_length=cfg['dataset']['seq_len'],
            date_aug_range=cfg['dataset']['data_aug_range'],
            sampling_mode=cfg['dataset']['sampling_mode'],
            domain_shift=True,
        )

    elif cfg['dataset']['name'].lower() == 'dynamicearthnetmsi':
        trainset = DynamicEarthNetMSI(
            path=cfg['dataset']['path'],
            split="train",
            num_channels=cfg['dataset']['num_channels'],
            num_classes=cfg['dataset']['num_classes'],
            img_size=cfg['dataset']['image_size'],
            true_size=cfg['dataset']['true_size'],
            train_length=cfg['dataset']['seq_len'],
            date_aug_range=cfg['dataset']['data_aug_range'],
            sampling_mode=cfg['dataset']['sampling_mode'],
        )
        testset_in = DynamicEarthNetMSI(
            path=cfg['dataset']['path'],
            split="val",
            num_channels=cfg['dataset']['num_channels'],
            num_classes=cfg['dataset']['num_classes'],
            img_size=cfg['dataset']['image_size'],
            true_size=cfg['dataset']['true_size'],
            train_length=cfg['dataset']['seq_len'],
            date_aug_range=cfg['dataset']['data_aug_range'],
            sampling_mode=cfg['dataset']['sampling_mode'],
            domain_shift=False,
        )
        testset_out = DynamicEarthNetMSI(
            path=cfg['dataset']['path'],
            split="val",
            num_channels=cfg['dataset']['num_channels'],
            num_classes=cfg['dataset']['num_classes'],
            img_size=cfg['dataset']['image_size'],
            true_size=cfg['dataset']['true_size'],
            train_length=cfg['dataset']['seq_len'],
            date_aug_range=cfg['dataset']['data_aug_range'],
            sampling_mode=cfg['dataset']['sampling_mode'],
            domain_shift=True,
        )

    else:
        raise ValueError('Unknown dataset: {}'.format(cfg['dataset']['name']))

    dataloader_module = ImageDataModule(
        train_dataset=trainset,
        val_dataset=testset_in,
        test_dataset=testset_in,
        global_batch_size=cfg['dataset']['global_batch_size'],
        num_workers=cfg['computer']['num_workers'],

    )
    dataloader_module.setup(stage="fit")
    train_loader = dataloader_module.get_train_loader()
    dataloader_module.setup(stage="test")
    test_loader_in = dataloader_module.get_test_loader()
    dataloader_module = ImageDataModule(
        train_dataset=trainset,
        val_dataset=testset_out,
        test_dataset=testset_out,
        global_batch_size=cfg['dataset']['global_batch_size'],
        num_workers=cfg['computer']['num_workers'],
    )
    dataloader_module.setup(stage="test")
    test_loader_out = dataloader_module.get_test_loader()


    return train_loader, test_loader_in, test_loader_out

def load_trainer(cfg, model, trainloader, testloader_in, testloader_out, save_ckpt_path, save_log_path, warmup=False):
    loss_fn = Losses(
        mix=cfg['loss']['mix'],
        ignore_index=cfg['loss']['ignore_index'],
    )

    warmup_epochs = cfg['trainer'].get('warmup_epochs', 0)
    total_epochs = cfg['trainer']['max_epochs']
    base_lr = cfg['trainer']['lr']
    eta_min = base_lr * 0.01

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=cfg['trainer']['weight_decay'])

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        else:
            cosine_epoch = current_epoch - warmup_epochs
            cosine_T = total_epochs - warmup_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * cosine_epoch / cosine_T))
            return eta_min / base_lr + (1.0 - eta_min / base_lr) * cosine_decay

    if warmup:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['trainer']['max_epochs'],
                                                               eta_min=eta_min, last_epoch=-1)

    metric_fn = SCDMetric(
        num_classes=cfg['dataset']['num_classes'],
        ignore_index=cfg['dataset']['ignore_index'],
        class_names=cfg['dataset']['class_names'],
    )

    trainer = SitsScdWrapper(
        model=model,
        loss_fn=loss_fn,
        val_metrics=metric_fn,
        trainloader=trainloader,
        testloader_in=testloader_in,
        testloader_out=testloader_out,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg['computer']['accelerator'],
        seq_len=cfg['dataset']['seq_len'],
        resume=cfg['trainer']['resume'],
        prev_ckpt=cfg['trainer']['prev_ckpt'],
        ckpt_interval=cfg['trainer']['ckpt_interval'],
        save_ckpt_path=save_ckpt_path,
        save_log_path=save_log_path,
        model_name=cfg['model']['name'],
    )
    ckpt_root = os.path.dirname(save_ckpt_path)
    clean_empty_ckpt_dirs(ckpt_root)
    visual_root = os.path.join(cfg['visualize']['dir'], cfg['dataset']['name'])
    clean_orphan_log_dirs(log_root=os.path.dirname(save_log_path), ckpt_root=ckpt_root, vis_root=visual_root)
    return trainer

def load_tester(cfg, model, testloader, visual_path, in_out, ckpt_path=None, log_txt_path=""):
    metric_fn = SCDMetric(
        num_classes=cfg['dataset']['num_classes'],
        ignore_index=cfg['dataset']['ignore_index'],
        class_names=cfg['dataset']['class_names'],
    )

    if ckpt_path is None: # take last time training
        ckpt_root = os.path.join(cfg['checkpoints']['dir'], cfg['dataset']['name'])
        ckpt_path = find_latest_ckpt(ckpt_root, filename=f'best_{in_out}.tar.pth')
    else: # specific ckpt path
        ckpt_path = os.path.join(ckpt_path, f'best_{in_out}.tar.pth')
    print("Test phase load ckpt from {}".format(ckpt_path))

    with open(log_txt_path, "a") as f:
        f.write("*" * 10 + f"{in_out.upper()} Domain Test phase load ckpt from {ckpt_path}" + "*" * 10 + "\n")
    clean_empty_visual_dirs(vis_root=os.path.dirname(visual_path))

    tester = Tester(
        data=cfg['dataset']['name'],
        model=model,
        metric_fn=metric_fn,
        testloader=testloader,
        visual_path=visual_path,
        device=cfg['computer']['accelerator'],
        ckpt_path=ckpt_path,
        log_txt_path=log_txt_path,
        model_name=cfg['model']['name'],
        in_out=in_out
    )
    return tester



def find_latest_ckpt(ckpt_root, filename="best.tar.pth"):
    subdirs = [
        d for d in os.listdir(ckpt_root)
        if os.path.isdir(os.path.join(ckpt_root, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {ckpt_root}")

    for subdir in sorted(subdirs, reverse=True):
        ckpt_path = os.path.join(ckpt_root, subdir, filename)
        if os.path.exists(ckpt_path):
            return ckpt_path

    raise FileNotFoundError(f"No '{filename}' found in any subdirectory of {ckpt_root}")



