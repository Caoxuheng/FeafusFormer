# [Unsupervised Hybrid Network of Transformer and CNN for Blind Hyperspectral and Multispectral Image Fusion](https://ieeexplore.ieee.org/abstract/document/10415455)
This is a transformer-based super-resolution algorithm that fuse hyperspectral and multispectral image in a unsupervised manner in coping with unknown degradation in spectral and spatial domain. 
# Flowchart
![Flowchart](https://github.com/Caoxuheng/imgs/blob/main/HIFtool/flowchart_Feafusformer.png)
# Result presentation  
Nonblind fusion results on Pavia, Chikusei and Xiongan datasets.  
![Result](https://github.com/Caoxuheng/imgs/blob/main/HIFtool/result_feafusformer.png)
The reconstructed results on CAVE can be downloaded from [`here`](https://aistudio.baidu.com/aistudio/datasetdetail/173277).
# Guidance
**None**
# Requirements
## Environment
`Python3.8`  
`torch 1.12`,`torchvision 0.13.0`  
`Numpy`,`Scipy`  
## Datasets
[`CAVE dataset`](https://www1.cs.columbia.edu/CAVE/databases/multispectral/), 
 [`Preprocessed CAVE dataset`](https://aistudio.baidu.com/aistudio/datasetdetail/147509).
# Note
For any questions, feel free to email me at caoxuheng@tongji.edu.com.  
If you find our work useful in your research, please cite our paper ^.^

# Citation
```python
@ARTICLE{10415455,
  author={Cao, Xuheng and Lian, Yusheng and Wang, Kaixuan and Ma, Chao and Xu, Xianqing},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Unsupervised Hybrid Network of Transformer and CNN for Blind Hyperspectral and Multispectral Image Fusion}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Degradation;Transformers;Spatial resolution;Imaging;Tensors;Spectral analysis;Hyperspectral imaging;Blind fusion;degradation representation;feature fusion;super-resolution;unsupervised transformer},
  doi={10.1109/TGRS.2024.3359232}}
