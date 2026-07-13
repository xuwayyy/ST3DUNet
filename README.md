#  Spatial-Temporal Collaborative Network for Satellite Image Time Series Semantic Change Detection

Official implementation of the paper **"Spatial-Temporal Collaborative Network for Satellite Image Time Series Semantic Change Detection"**, published in **IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2026**.

---

# 🔧 Environment Setup

We recommend using **Python 3.10**, **PyTorch 2.3.0**, and **CUDA 12.1**.

## 1. Create the Conda Environment

```bash
conda create -n st3dunet python=3.10 -y
conda activate st3dunet
```

## 2. Install Dependencies

First, install **PyTorch 2.3.0** with **CUDA 12.1** following the official PyTorch installation guide.

Then install the remaining dependencies:

```text
pytorch==2.3.0
rasterio==1.4.3
opencv-python==4.12.0
numpy==1.26.4
```

---

# 📦 Dataset Preparation

We use the pre-processed versions of the **DynamicEarthNet** and **MUDS** datasets provided by the [SitsSCD](https://github.com/ElliotVincent/SitsSCD) project. The pre-processing includes image compression to improve memory efficiency during long-term time-series training.

Download the datasets from the following links:

- **DynamicEarthNet**  
  https://drive.google.com/file/d/1cMP57SPQWYKMy8X60iK217C28RFBkd2z/view?usp=drive_link

- **MUDS**  
  https://drive.google.com/file/d/1RySuzHgQDSgHSw2cbriceY5gMqTsCs8I/view?usp=drive_link

- **DynamicEarthNetMSI**
  https://dataserv.ub.tum.de/index.php/s/m1650201

After downloading, extract the datasets into the `datasets/` directory.

---

# 🚀 Training

To train ST3DUNet from scratch (using **MUDS** as an example), configure the hyperparameters in `configs/muds.yaml` and run:

```bash
python main.py --data muds
```

You can also override configuration options directly from the command line. For example, to change the learning rate:

```bash
python main.py --data muds --lr 0.001
```

## Resume Training

To resume training from a saved checkpoint, use the `--resume` and `--resume_ckpt` arguments:

```bash
python main.py --data muds \
    --resume \
    --resume_ckpt checkpoints/muds/your_checkpoint.pth
```

---

# 📊 Evaluation

To evaluate a trained model, simply add the `--test` flag:

```bash
python main.py --data muds --test
```

By default, the latest experiment folder corresponding to the model name in YAML will be located automatically, and the checkpoint inside will be loaded for evaluation.

To evaluate a specific checkpoint, modify the checkpoint path in **`main.py` (around Line 106)** and run the same command above.

---

# 🙏 Acknowledgements

This project benefits from the pre-processed **DynamicEarthNet** and **MUDS** datasets provided by the authors of [SitsSCD](https://github.com/ElliotVincent/SitsSCD). We gratefully acknowledge their effort in preparing and releasing these resources.

