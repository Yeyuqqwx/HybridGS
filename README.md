# HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting
This repository contains the code for the paper [HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting](https://gujiaqivadin.github.io/hybridgs/).
![](https://gujiaqivadin.github.io/hybridgs/static/images/1.jpg)
Currently, only the inference part of the code is open sourced to demonstrate some implementation details and effects. The training part of the code is not open sourced yet.
The current implementation is based on Taming-3DGS, gsplat and Gaussianimage, and more implementation details will be open sourced step by step in the future.
## Requirements
Follow the instructions in the [Taming 3DGS repository](https://github.com/humansensinglab/taming-3dgs) and [GaussianImage repository](https://github.com/Xinjie-Q/GaussianImage)  to setup the environment. Or simply run the following commands:
```bash
cd src/gimage/gaussianimage/gsplat && python setup.py build_ext --inplace -j32
cp build/*/*/csrc.so gsplat/
```



```bash
git clone https://github.com/humansensinglab/taming-3dgs.git --recursive
cd taming-3dgs
pip install submodules/*
```

## Dataset and Checkpoints
We use the following datasets for evaluation:

[NeRF On-the-go dataset](https://huggingface.co/datasets/jkulhanek/nerfonthego-wg/tree/main)

[RobustNeRF dataset](https://storage.googleapis.com/jax3d-public/projects/robustnerf/robustnerf.tar.gz)

Please modify the `DATA_DIR` in `render_and_metric.sh` to the path of your dataset.

We provide the checkpoints for rendering and evaluation in [HybridGS checkpoints](https://huggingface.co/Eto63277/HybridGS/tree/main).

Please modify the `CKPT_PATH` in `render_and_metric.sh` to the path of your checkpoints.

## Usage

```bash
./render_and_metric.sh
```

## Citation
If you find our code or paper useful, please cite:
```bibtex
@InProceedings{lin2024hybridgs,
    author = {Jingyu Lin, Jiaqi Gu, Lubin Fan, Bojian Wu, Yujing Lou, Renjie Chen, Ligang Liu, Jieping Ye},
    title = {HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting},
    booktitle = {Arxiv},
    year = {2024}
}
```
