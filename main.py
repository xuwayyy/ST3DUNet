import os
from pathlib import Path
import argparse
from datetime import datetime
import yaml
from utils.wrapper import load_yaml, load_model, load_loader, load_trainer, load_tester


def parse_args():
    parser = argparse.ArgumentParser(description='Times Series Remote Sensing Semantic Change Detection')
    # data
    parser.add_argument('--data', type=str, default='muds',
                        help='dynamicearthnet, muds, h2crop')
    # training
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--cosine", action="store_false", help="use cosine annealing")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--resume", action="store_true", help="resume training default false")
    parser.add_argument("--ckpt", type=Path, help="ckpt path", default="")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")

    return parser.parse_args()

def seed_all(seed):
    import random
    import numpy as np
    import torch
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
        'prev_ckpt': str(args.ckpt) if str(args.ckpt) != "" else None,
    })
    cfg['trainer'] = trainer_cfg

    seed_all(cfg['run']['seed'])


    print(cfg)
    trainloader, testloader_in, testloader_out = load_loader(cfg)


    model = load_model(cfg)


    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {params / 1e6:.2f} M")

    log_txt_path = os.path.join(save_log_path, "log.txt")
    with open(log_txt_path, "w") as f:
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

    ckpt_path = None # For Testing, None uses latest, or specifc to a ckpt path

    tester_in = load_tester(cfg, model, testloader_in, in_out="in",  visual_path=save_visual_path,
                            ckpt_path=ckpt_path, log_txt_path=log_txt_path)

    print("In Domain Test Begin!!!!")
    tester_in.test()

    tester_out = load_tester(cfg, model, testloader_out, in_out="out",  visual_path=save_visual_path,
                             ckpt_path=ckpt_path, log_txt_path=log_txt_path)

    print("Out Domain Test Begin!!!!")
    tester_out.test()

if __name__ == '__main__':
    main()