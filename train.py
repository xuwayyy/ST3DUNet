import os
from tqdm import tqdm
import torch
import torch.nn as nn
from collections import defaultdict

class SitsScdWrapper(nn.Module):
    def __init__(self, model, loss_fn, val_metrics,
                 trainloader, testloader_in, testloader_out ,optimizer, scheduler=None,device="gpu", seq_len=8,
                 resume=False, prev_ckpt=None, ckpt_interval=10, save_ckpt_path="",
                 save_log_path="", model_name="st3dunet"):
        super().__init__()
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.metric_fn = val_metrics
        self.device = device
        self.trainloader = trainloader
        self.testloader_in = testloader_in
        self.testloader_out = testloader_out
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_epoch = 1
        self.ckpt_interval = ckpt_interval
        self.save_ckpt_path = save_ckpt_path
        self.save_log_path = save_log_path
        self.seq_len = seq_len
        self.model_name = model_name

        keys = ["acc", "macc", "miou", "bc", "sc", "scs"]

        self.best_score_in = {k: 0 for k in keys}
        self.best_score_out = {k: 0 for k in keys}
        if resume and prev_ckpt is not None:
            ckpt = torch.load(prev_ckpt)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_epoch = ckpt['epoch'] + 1
            self.best_score_in = ckpt['best_score_in']
            self.best_score_out = ckpt['best_score_out']
            self.scheduler.load_state_dict(ckpt['scheduler'])
            print(f"Resume Training from epoch f{self.start_epoch}")

    def compute_loss(self, pred, batch, average=True):
        if not isinstance(pred, dict):
            pred = {"logits": pred}
        if not isinstance(batch, dict):
            batch = {"gt": batch}
        return self.loss_fn(pred, batch, average=average)

    def train_one_epoch(self, epoch=None):
        self.model.train()
        pbar = tqdm(self.trainloader, desc="🟢 Training", colour="green")
        total_steps = len(self.trainloader)
        step_log_interval = max(1, total_steps // 3)

        epoch_loss_sums = defaultdict(float)

        for step, batch in enumerate(pbar):
            # every value in batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


            x = batch['data'].to(self.device)
            y = batch['gt'].to(self.device)
            date = batch['positions'].to(self.device)
            logits = self.model(x, date)
            losses = self.compute_loss(pred=logits, batch=y, average=True)
            loss = losses["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            current_step = epoch * total_steps + step if epoch is not None else step

            for name, val in losses.items():
                val_item = val.item()
                epoch_loss_sums[name] += val_item

            if (step + 1) % step_log_interval == 0:
                loss_log = ", ".join([f"{name}: {val.item():.4f}" for name, val in losses.items()])
                print(f"[Epoch {epoch}] Step [{step + 1}/{total_steps}], {loss_log}")

            pbar.set_postfix(loss=loss.item())

        avg_losses = {name: total / len(self.trainloader) for name, total in epoch_loss_sums.items()}

        return avg_losses

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        self.metric_fn.reset()
        log_txt_path = os.path.join(self.save_log_path, 'log.txt')
        pbar = tqdm(self.testloader_in, desc="🔵 Evaluating In Domain", colour="blue")
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            x = batch['data'].to(self.device)
            y = batch['gt'].to(self.device)
            date = batch['positions'].to(self.device)
            logits = self.model(x, date)
            pred = logits.argmax(2).view_as(y)
            self.metric_fn.update(pred.cpu(), y.cpu())

        metrics = self.metric_fn.compute()
        print("*" * 25)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("*" * 25)

        with open(log_txt_path, 'a') as f:
            f.write(f"[In Domain Eval @ Epoch {epoch}]\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("=" * 50 + '\n')

        miou, scs = metrics['miou'], metrics['scs']
        if miou > self.best_score_in['miou']:
            self.save_model(save_root_path=self.save_ckpt_path, metric=metrics, epoch=epoch, intermediate=True, in_out="in")

        pbar = tqdm(self.testloader_out, desc="🔵 Evaluating Out Domain", colour="yellow")
        self.metric_fn.reset()
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            x = batch['data'].to(self.device)
            y = batch['gt'].to(self.device)
            date = batch['positions'].to(self.device)
            logits = self.model(x, date)

            pred = logits.argmax(2).view_as(y)
            self.metric_fn.update(pred.cpu(), y.cpu())

        metrics = self.metric_fn.compute()
        print("*" * 25)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("*" * 25)

        with open(log_txt_path, 'a') as f:
            f.write(f"[Out Domain Eval @ Epoch {epoch}]\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("=" * 50 + '\n')

        miou, scs = metrics['miou'], metrics['scs']
        if miou > self.best_score_out['miou']:
            self.save_model(save_root_path=self.save_ckpt_path, metric=metrics, epoch=epoch, intermediate=True, in_out="out")



    def save_model(self, save_root_path, in_out, epoch, metric=None, intermediate=True,):
        assert in_out in ['in', 'out']
        if intermediate:
            assert metric is not None, 'save_intermediate_best must provide current metric to update'
            if in_out == 'in':
                self.best_score_in = metric
            elif in_out == 'out':
                self.best_score_out = metric
            ckpt = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_score_in': self.best_score_in,
                'best_score_out': self.best_score_out,
                'epoch': epoch,
            }
            save_path = os.path.join(save_root_path, f"best_{in_out}.tar.pth")
            print(f"✅ saved In Domain Best model to {save_path} at epoch {epoch}")
        else:
            ckpt = {
                'model': self.model.state_dict(),
                'best_score_in': self.best_score_in,
                'best_score_out': self.best_score_out,
                'epoch': epoch,
            }
            save_path = os.path.join(save_root_path, f"epoch{epoch}.tar.pth")
            print(f"🔧 saved ckpt inverval model to {save_path} at epoch {epoch}")
        torch.save(ckpt, save_path)

    def train_epochs(self, epochs,):
        log_txt_path = os.path.join(self.save_log_path, 'log.txt')
        for epoch in range(self.start_epoch, epochs + self.start_epoch):
            epoch_losses = self.train_one_epoch(epoch)
            loss_log = ", ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()])
            print(f"Epoch {epoch} - {loss_log}")

            self.evaluate(epoch=epoch)
            if epoch % self.ckpt_interval == 0 or epoch == epochs + self.start_epoch - 1:
                self.save_model(save_root_path=self.save_ckpt_path, epoch=epoch, intermediate=False)

            with open(log_txt_path, 'a') as f:
                f.write(f"[Epoch {epoch}] Train Losses: {loss_log}\n")
                f.write("          Best Scores:\n")
                for k, v in self.best_score.items():
                    f.write(f"            {k}: {v:.4f}\n")
                f.write("=" * 50 + '\n')


if __name__ == "__main__":
    pass