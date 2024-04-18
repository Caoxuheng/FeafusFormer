<img align="right" src="https://ieeexplore.ieee.org/ielx7/36/9624468/9625588.jpg" />  

# [Unsupervised Hybrid Network of Transformer and CNN for Blind Hyperspectral and Multispectral Image Fusion](https://ieeexplore.ieee.org/abstract/document/10415455)
Fusing a low spatial resolution hyperspectral image (LR-HIS) with a high spatial resolution multispectral image has become popular for generating a high spatial resolution hyperspectral image (HR-HSI). Most methods assume that the degradation information from high resolution to low resolution is known in spatial and spectral domains. Conversely, this information is often limited or unavailable in practice, restricting their performance. Furthermore, existing fusion methods still face the problem of insufficient exploration of the cross-interaction between the spatial and spectral domains in the HR-HSI, leaving scope for further improvement. This article proposes an unsupervised hybrid network of transformer and CNN (uHNTC) for blind HSI-MSI fusion. The uHNTC comprises three subnetworks: a transformer-based feature fusion subnetwork (FeafusFomer) and two CNN-based degradation subnetworks (SpaDNet and SpeDNet). Considering the strong multilevel spatio-spectral correlation between the desired HR-HSI and the observed images, we design a multilevel cross-feature attention (MCA) mechanism in FeafusFormer. By incorporating the hierarchical spatio-spectral feature fusion into the attention mechanism in the transformer, the MCA globally keeps a high spatio-spectral cross-similarity between the recovered HR-HSI and observed images, thereby ensuring the high cross-interaction of the recovered HR-HSI. Subsequently, the characteristics of degradation information are utilized to guide the design of the SpaDNet and SpeDNet, which helps FeafusFormer accurately recover the desired HR-HSI in complex real-world environments. Through an unsupervised joint training of the three subnetworks, uHNTC recovers the desired HR-HSI without preknown degradation information. Experimental results on three public datasets and a WorldView-2 images show that the uHNTC outperforms ten state-of-the-art fusion methods.   
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
  volume={62},
  number={},
  pages={1-15},
  keywords={Degradation;Transformers;Spatial resolution;Imaging;Tensors;Spectral analysis;Hyperspectral imaging;Blind fusion;degradation representation;feature fusion;superresolution;unsupervised transformer},
  doi={10.1109/TGRS.2024.3359232}}
