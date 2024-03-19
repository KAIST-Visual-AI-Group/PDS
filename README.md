# Posterior Distillation Sampling
![teaser](./assets/teaser.png)

[**arXiv**](https://arxiv.org/abs/2311.13831) | [**Project Page**](https://posterior-distillation-sampling.github.io/) <br>

[Juil Koo](https://63days.github.io), [Chanho Park](https://github.com/charlieppark), [Minhyuk Sung](https://mhsung.github.io/) <br>

# Introduction
This repository contains the official implementation of **Posterior Distillation Sampling (PDS)**. <br>
PDS enables various parametric image editing, including NeRF, 3D Gaussian Splatting and SVG editing. More results can be viewed at our [project webpage](https://posterior-distillation-sampling.github.io/).

# UPDATES
- [x] PDS core code.
- [x] NeRF editing code.
- [x] 3D Gaussian Splatting editing code.
- [x] SVG editing code.

[//]: # (### Abstract)
> We introduce Posterior Distillation Sampling (PDS), a novel optimization method for parametric image editing based on diffusion models. Existing optimization-based methods, which leverage the powerful 2D prior of diffusion models to handle various parametric images, have mainly focused on generation. Unlike generation, editing requires a balance between conforming to the target attribute and preserving the identity of the source content. Recent 2D image editing methods have achieved this balance by leveraging the stochastic latent encoded in the generative process of diffusion models. To extend the editing capabilities of diffusion models shown in pixel space to parameter space, we reformulate the 2D image editing method into an optimization form named PDS. PDS matches the stochastic latents of the source and the target, enabling the sampling of targets in diverse parameter spaces that align with a desired attribute while maintaining the source's identity. We demonstrate that this optimization resembles running a generative process with the target attribute, but aligning this process with the trajectory of the source's generative process. Extensive editing results in Neural Radiance Fields and Scalable Vector Graphics representations demonstrate that PDS is capable of sampling targets to fulfill the aforementioned balance across various parameter spaces.

# Get Started

## Construct a Conda Environment

Before running 3D and SVG editing with PDS, first build a conda environment and install the `pds` library:
```
git clone https://github.com/KAIST-Visual-AI-Group/PDS
cd PDS
conda create -n pds python=3.9
conda activate pds
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -e .
```

## 3D Editing
For 3D editing, including NeRF and [3D Gaussian Splatting (3DGS)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) editing, refer to [`3d_editing/README.md`](./3d_editing/README.md).

## SVG Editing
For SVG editing, refer to [`svg_editing/README.md`](./svg_editing/README.md).

# Citation
If you find our work useful, please consider citing:
```
@inproceedings{Koo:2024PDS,
    title={Posterior Distillation Sampling},
    author={Koo, Juil and Park, Chanho and Sung, Minhyuk},
    booktitle={CVPR},
    year={2024}
}
```

