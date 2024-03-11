# AdaBLDM
The implement for paper : "A Novel Approach to Industrial Defect Generation through Blended Latent Diffusion Model with Online Adaptation" 

[Arxiv Link](https://arxiv.org/abs/2402.19330)




## Dataset
1. [MVTEC AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
2. [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip)
3. [KSDD2](https://www.vicos.si/resources/kolektorsdd2/)
    > Due to the non-square ksdd2 image   , it need to crop the image of KSDD2 samples. 
    >
    >**Preprocess code in** [this](./prepare).

## Prepare
> Build trainset for AdaBLDM training. 

**MVTec-AD** in [this](./prepare//mvtec/readme.md).

**BTAD** in [this](./prepare//btad/readme.md).

**KSDD** in [this](./prepare/btad/readme.md).

## Train
Coming soon....

## Stable Diffusion 

**Pretrain Stage**

How to obtain the mvtec object's description?
[Look at this.](https://github.com/GrandpaXun242/Img2Text)

**Foreground_predictor for trimap**

How to get the object's foreground ?
[Look at this](./foreground_predictor/)

## SVM
Coming soon....
## DeSTSeg(CVPR2023)
DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection

[offical code](https://github.com/apple/ml-destseg)

## Compare Method
### DFMGAN(AAAI2023)
Few-Shot Defect Image Generation via Defect-Aware Feature Manipulation

[offical code](https://github.com/Ldhlwh/DFMGAN)

## Eval metric
**Anomaly detection (following DeSTSeg)**
1. pixel-auc
2. pro
3. ap
4. iap
5. iap90

**Image quality (following DFMGAN)**
1. KID
2. LPIPS

## Citation
```shell
@article{Li2024ANA,
    title={A Novel Approach to Industrial Defect Generation through Blended Latent Diffusion Model with Online Adaptation},
    author={Hanxi Li and Zhengxun Zhang and Hao Chen and Lin Wu and Bo Li and Deyin Liu and Mingwen Wang},
    year={2024},
    url={https://api.semanticscholar.org/CorpusID:268091266},
}
```





