import torch
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

platte_muds = [
    [0, 0, 0], # class 0: black
    [255, 255, 255] # class 1: white
]

platte_dynamic = [
    [96, 96, 96], # imperv surface
    [204, 204, 0],  #  agriculture
    [0, 204, 0], # forest & other
    [0, 0, 153], # wetlands
    [153, 76, 0], # soil
    [0, 127, 255], # water
 ]


def decode_segmap(mask, palette):
    """Decode segmentation mask to RGB image using palette"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_mask[mask == label] = color
    return color_mask


def save_img(path, array, cmap=None, size=(512, 512), interp="bicubic"):

    interp_map = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS
    }
    if array.ndim == 2:  # 灰度
        if cmap:
            import matplotlib.cm as cm
            array = (cm.get_cmap(cmap)(array / 255.0)[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(array)
    else:
        img = Image.fromarray(array)

    img = img.resize(size, interp_map.get(interp, Image.BICUBIC))
    img.save(path)

def save_T_tensor_to_img(
    x, gt, pred, data, visual_path, mean, std):

    semantic_dir = os.path.join(visual_path, "semantic")
    change_dir = os.path.join(visual_path, "change")
    os.makedirs(semantic_dir, exist_ok=True)
    os.makedirs(change_dir, exist_ok=True)

    palette = platte_muds if data.lower() == "muds" else platte_dynamic

    x = x.squeeze(0).cpu().numpy()        # T x C x H x W
    gt = gt.squeeze(0).cpu().numpy()      # T x H x W
    pred = pred.squeeze(0).cpu().numpy()  # T x H x W

    x = x.transpose(0, 2, 3, 1)  # T x H x W x C

    for t in range(x.shape[0]):
        x_i = x[t] * std.numpy().transpose(1, 2, 0) + mean.numpy().transpose(1, 2, 0)
        x_i = np.clip(x_i, 0, 255).astype(np.uint8)
        if x_i.shape[-1] == 4 :
            x_i = x_i[..., :3]
        elif x_i.shape[-1] == 12:
            r = x_i[..., 0]
            g = x_i[..., 2]
            b = x_i[..., 1]
            x_i = np.stack([r, g, b], axis=-1)

        gt_i = gt[t]
        pred_i = pred[t]

        gt_color = decode_segmap(gt_i, palette)
        pred_color = decode_segmap(pred_i, palette)
        image_size = 256
        save_img(os.path.join(semantic_dir, f"frame_{t}_input.png"), x_i, size=(image_size, image_size), interp='bicubic')
        save_img(os.path.join(semantic_dir, f"frame_{t}_gt.png"), gt_color, size=(image_size, image_size), interp='nearest')
        save_img(os.path.join(semantic_dir, f"frame_{t}_pred.png"), pred_color, size=(image_size, image_size), interp='nearest')


        if t < x.shape[0] - 1:
            cd_gt = (gt[t] != gt[t + 1]).astype(np.uint8) * 255
            cd_pred = (pred[t] != pred[t + 1]).astype(np.uint8) * 255

            save_img(os.path.join(change_dir, f"cd_frame_{t}_gt.png"), cd_gt, cmap="gray", size=(image_size, image_size), interp='nearest')
            save_img(os.path.join(change_dir, f"cd_frame_{t}_pred.png"), cd_pred, cmap="gray", size=(image_size, image_size), interp='nearest')


class Tester:
    def __init__(self, data, model, testloader, ckpt_path:str, visual_path, metric_fn,
                 log_txt_path, model_name, in_out, device="gpu" ):
        self.data = data
        self.model = model.to(device)
        self.testloader = testloader
        print("Loading CKPT from {}".format(ckpt_path))
        self.ckpt = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(self.ckpt["model"]) if "model" in self.ckpt else self.model.load_state_dict(self.ckpt)
        if in_out in ["in", "out"]:
            self.visual_path = os.path.join(visual_path, "no domain shift" if in_out == "in" else "spatial domain shift")
        else: # val, test
            self.visual_path = os.path.join(visual_path, f"{in_out}")
        self.metric_fn = metric_fn
        self.device = device
        self.log_txt_path = log_txt_path
        self.model_name = model_name
        self.in_out = in_out
        if self.data.lower() == 'muds':
            self.mean = torch.tensor([119.9347, 105.3608, 77.5125], dtype=torch.float32).reshape(3, 1, 1)
            self.std = torch.tensor([59.5921, 48.2708, 44.7296], dtype=torch.float32).reshape(3, 1, 1)
        elif self.data.lower() == 'dynamicearthnet':
            if in_out in ["in", "out"]:
                self.mean = torch.tensor([83.1029, 80.7615, 69.3328, 133.8648], dtype=torch.float32).reshape(4, 1, 1)
                self.std = torch.tensor([33.2714, 25.5288, 23.9868, 30.5591], dtype=torch.float32).reshape(4, 1, 1)
            else:
                self.mean = torch.tensor([1042.59, 915.62, 671.26, 2605.21], dtype=torch.float32).reshape(4, 1, 1)
                self.std = torch.tensor([957.96, 715.55, 596.94, 1059.90], dtype=torch.float32).reshape(4, 1, 1)
        elif self.data.lower() == 'dynamicearthnetmsi':
            self.mean = torch.tensor([1161.5256, 1399.3857, 1455.7269, 2761.0648, 1815.2145, 2465.5625,
                                      2722.3400, 2867.8205, 2336.8220, 1742.1495,
                                      1069.3466, 3128.7744, ], dtype=torch.float32).reshape(12, 1, 1)
            self.std = torch.tensor([1541.3866, 1459.2290, 1528.1924, 1405.8515, 1518.5338, 1386.2684,
                                     1405.3624, 1410.5013, 1377.1001, 1296.2761, 1609.3568, 2166.7642],
                                    dtype=torch.float32).reshape(12, 1, 1)

        if self.ckpt.get('best_score') is not None and self.ckpt.get('best_score_in') is None:
            print("Training Phase CKPT In Domain Best score: ", self.ckpt["best_score"])
        elif self.ckpt.get('best_score_in') is not None or self.ckpt.get('best_score_out') is not None:
                print(f"Training Phase CKPT {in_out.upper()} Domain Best score: ", self.ckpt[f"best_score_{in_out}"])

        elif self.ckpt.get('miou') is not None:
            print(f"Training Phase CKPT {in_out.upper()} Best mIoU ", self.ckpt["miou"])

    def test(self):
        self.model.eval()
        self.metric_fn.reset()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.testloader)):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                x = batch['data']
                y = batch['gt']
                date = batch['positions']
                logits = self.model(x, date)
                pred = logits.argmax(2).view_as(y)


                self.metric_fn.update(pred.cpu(), y.cpu())

                save_T_tensor_to_img(
                    x.detach().cpu(),
                    y.detach().cpu(),
                    pred.detach().cpu(),
                    self.data,
                    visual_path=os.path.join(self.visual_path, f"sample_{i}"),
                    mean=self.mean,
                    std=self.std,)

            metrics = self.metric_fn.compute()
            print("*" * 25)
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            print("*" * 25)
            with open(self.log_txt_path, "a") as f:
                f.write(f"\n=========={self.in_out.upper()} Domain Test Results ==========\n")
                for k, v in metrics.items():
                    f.write(f"{k}: {v:.4f}\n")
                f.write("==================================\n")
