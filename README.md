# Spatial-Temporal Collaborative Network for Satellite Image Time Series Semantic Change Detection
This is the repository of our paper **Spatial-Temporal Collaborative Network for Satellite Image Time Series Semantic Change Detection** published in TGRS 2026

## Environment Setup

Recommended **Python = 3.10** and **Pytorch=2.3.0** 

### Required Dependencies

```text
pytorch==2.2.0
rasterio==1.4.3
opencv-python==4.12.0
numpy==1.26.4
```
### Installation Example
```
conda create -n st3dunet python=3.10 -y
conda activate st3dunet
```
Install ```pytorch=2.3.0, cuda 12.1``` version from pytorch org first and install other essential resources

## Dataset Preparation
We use [SitsSCD](https://github.com/ElliotVincent/SitsSCD) pre-processed data for DynamicEarthNet and Muds, the pre-processing consists in image compression for memory efficiency. You can download the datasets using the code below or by following these links for [DynamicEarthNet](https://drive.google.com/file/d/1cMP57SPQWYKMy8X60iK217C28RFBkd2z/view?usp=drive_link) and [Muds](https://drive.google.com/file/d/1RySuzHgQDSgHSw2cbriceY5gMqTsCs8I/view?usp=drive_link)

After downloading, place the data under ```datasets``` folder

## Train ST3DUNet
Example in Muds, complete ```muds.yaml```, and run 
```
python main.py --data muds
```
Or you can overwrite some parameters like learning rate in yaml by 
```
python main.py --data muds --lr 0.001
```

## Test ST3DUNet
Example in Muds, the code will automatic search the latest checkpoint according to your model name in yaml
```
python main.py --data muds --test
```
Or specific to a checkpoint by changing the **main.py** line 106 to a folder and run above command


