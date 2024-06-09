# Point-Cloud Completion with Pretrained Text-to-image Diffusion Models
### [Project Page](https://sds-complete.github.io/) | [Paper](https://arxiv.org/pdf/2306.10533.pdf) 


This repository contains the official code implementation for SDS-Complete (NeurIPS 2023). 
The code is based on https://github.com/ashawkey/stable-dreamfusion. 



## Installation Requirements
The code is compatible with Python 3.7 and pytoch 1.13.1. We recommned using ancaonda and pip to install the required packages:
```
conda create -n "sdscomplete" python=3.7
conda activate sdscomplete

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install scipy
pip install tqdm
pip install imageio
pip  install pandas
pip install scikit-image==0.18.3
pip install opencv-python
pip install matplotlib
pip install trimesh
pip install transformers
pip install diffusers


```

## Folder Structure
The code assumes that the folder data_processing/redwood_dataset contains the input scans (and GT surfaces for evaluation if available).  
```
.
├── main.py
├── ...
├── data_processing                    
│   ├── README_data.md
|   ├── ...      
│   └── redwood_dataset        
|       ├── depths
|       ├── GT
|       ├── point_clouds
|       └── world_planes            
└── workspace

```
See data_processing/README_data.md  for data processing instructions.


## Running example: 

```
python main.py --object_id_number=09639
```
A running folder with checkpoints, surfaces and rendering images will be logged to the folder workspace


## Citation
If you find our work useful in your research, please consider citing:
```
@article{kasten2024point,
  title={Point Cloud Completion with Pretrained Text-to-Image Diffusion Models},
  author={Kasten, Yoni and Rahamim, Ohad and Chechik, Gal},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```