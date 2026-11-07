# Beyond Stabilization: Dual-EMA Teachers for Global–Local Semantic Learning in Semi-Supervised Medical Image Segmentation

Official code for "Beyond Stabilization: Dual-EMA Teachers for Global–Local Semantic Learning in Semi-Supervised Medical Image Segmentation". (NeurIPS 2026)


---

## 1. Requirements
This repository is based on PyTorch 1.10.1, CUDA 11.1 and Python 3.6. 

First, install PyTorch separately:

```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Then install the remaining dependencies:

```
pip install -r requirements.txt
```

## 2. Usage
We provide code and data for LA  datasets.

Source data can be obtained from https://pan.baidu.com/s/1CSmBWL_VobZLnT_z0i4eQg?pwd=v3yx 
You can download directly, and move them to the folder `Datasets/la/data`.

To train a model,
```
python train_LA.py
```
To test your trained model, and get the final performance,
```
python test_LA.py
```

## Acknowledgements
This implementation builds upon the SSL4MIS codebase (https://github.com/HiLab-git/SSL4MIS). We sincerely thank the authors for making their work publicly available.

## Citation
If you use our code or datas in this repository for your research, please cite our papers.
```latex
@inproceedings{NeurIPS2026DualEMA-MedSeg,
  author = {Kaibo Liao, Yazhou Ren, Song Wu, Kecheng Chen, Xiaorong Pu, Lixin Duan},
  title = {Beyond Stabilization: Dual-EMA Teachers for Global–Local Semantic Learning in Semi-Supervised Medical Image Segmentation},
  journal ={40th Conference on Neural Information Processing Systems (NeurIPS 2026)},
  year = {2026},
}