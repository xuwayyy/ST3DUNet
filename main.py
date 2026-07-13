import os
from pathlib import Path
import argparse
from datetime import datetime
import yaml
from utils.wrapper import load_yaml, load_model, load_loader, load_trainer, load_tester
import random
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Times Series Remote Sensing Semantic Change Detection')
    # data
    parser.add_argument('--data', type=str, default='muds',
                        help='dynamicearthnet, muds, dynamicearthnetmsi')
    # training
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--cosine", action="store_false", help="use cosine annealing")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--resume", action="store_true", help="resume training default false")
    parser.add_argument("--resume_ckpt", type=Path, help="ckpt path for resume", default="")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")

    parser.add_argument('--test', action='store_true', help='run inference/test mode only')

    return parser.parse_args()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = parse_args()
    dataset_name = args.data
    cfg = load_yaml(f"configs/{dataset_name}.yaml")

    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")  # 月-日_时_分_秒
    save_ckpt_path = os.path.join(cfg['checkpoints']['dir'], cfg['dataset']['name'] , "_" + timestamp + cfg['model']['name'])
    save_log_path = os.path.join(cfg['logger']['dir'], cfg['dataset']['name'] , "_" + timestamp  + cfg['model']['name'])
    save_visual_path = os.path.join(cfg['visualize']['dir'], cfg['dataset']['name'] , "_" + timestamp  + cfg['model']['name'])

    os.makedirs(save_ckpt_path, exist_ok=True)
    os.makedirs(save_log_path, exist_ok=True)
    os.makedirs(save_visual_path, exist_ok=True)

    trainer_cfg = cfg.get('trainer', {})
    trainer_cfg.update({
        'lr': args.lr,
        'cosine': args.cosine,
        'weight_decay': args.weight_decay,
        'resume': args.resume,
        'prev_ckpt': str(args.resume_ckpt) if str(args.resume_ckpt) != "" else None,
    })
    cfg['trainer'] = trainer_cfg

    seed_all(cfg['run']['seed'])

    print(cfg)
    trainloader, testloader_in, testloader_out = load_loader(cfg)

    model = load_model(cfg)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {params / 1e6:.2f} M")

    log_txt_path = os.path.join(save_log_path, "log.txt")

    if not args.test:
        print("======>>> [TRAIN MODE] Started <<<======")

        with open(log_txt_path, "w") as f:
            f.write(f"Mode: TRAIN\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Model: {cfg.get('model', {}).get('name', 'unknown')}\n")
            f.write(f"Num Params: {params / 1e6:.2f} M\n\n")

            f.write("Hyperparameters:\n")
            for k in ['lr', 'cosine', 'weight_decay', 'resume']:
                f.write(f"  {k}: {cfg['trainer'][k]}\n")
            f.write(f"  seed: {cfg['run']['seed']}\n\n")

            f.write("Config YAML:\n")
            f.write(yaml.dump(cfg, sort_keys=False))
            f.write("\nCommand:\n")
            f.write(" ".join(os.sys.argv))

        trainer = load_trainer(cfg, model, trainloader,
                               testloader_in=testloader_in, testloader_out=testloader_out,
                               save_ckpt_path=save_ckpt_path, save_log_path=save_log_path)

        trainer.train_epochs(epochs=cfg['trainer']['max_epochs'])
        print("======>>> [TRAIN MODE] Finished Successfully <<<======")

    else:
        print("======>>> [TEST MODE] Started <<<======")

        ckpt_path = None # None will search latest ckpt according to config model name, or specific to a folder (not file)

        with open(log_txt_path, "a") as f:
            f.write(f"\nMode: TEST\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Evaluated Checkpoint: Auto (using latest via load_tester)\n")
            f.write("Command:\n")
            f.write(" ".join(os.sys.argv) + "\n")

        print("--> In Domain Test Begin!!!!")
        tester_in = load_tester(cfg, model, testloader_in, in_out="in", visual_path=save_visual_path,
                                ckpt_path=ckpt_path, log_txt_path=log_txt_path)
        tester_in.test()

        print("--> Out Domain Test Begin!!!!")
        tester_out = load_tester(cfg, model, testloader_out, in_out="out", visual_path=save_visual_path,
                                 ckpt_path=ckpt_path, log_txt_path=log_txt_path)
        tester_out.test()

        print("======>>> [TEST MODE] Finished <<<======")


if __name__ == '__main__':
    main()