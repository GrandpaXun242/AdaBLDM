# AdaBLDM
The implement for paper : "A Novel Approach to Industrial Defect Generation through Blended Latent Diffusion Model with Online Adaptation" 

[Arxiv Link](https://arxiv.org/abs/2402.19330)

# News!
**2024/04/25** : Update the training code!  :blush:



## Dataset
1. [MVTEC AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
2. [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip)
3. [KSDD2](https://www.vicos.si/resources/kolektorsdd2/)
    > Due to the non-square ksdd2 image   , it need to crop the image of KSDD2 samples. 
    >
    >**Preprocess code in** [this](./prepare).

## Env Install
We follows [ControlNet](https://github.com/lllyasviel/ControlNet/blob/main/README.md?plain=1#L63-L64).

## Prepare
> Build trainset for AdaBLDM training. 

**MVTec-AD** in [this](./prepare//mvtec/readme.md).

**BTAD** in [this](./prepare//btad/readme.md).

**KSDD** in [this](./prepare/btad/readme.md).

## Train
> Mvtec-AD Part
0. Download Mvtec-AD dataset.
1. **Foreground_predictor** : [Look this](./foreground_predictor//Tutorial.md)
2. **Prepare for mvtec dataset** : [Look this](./prepare/mvtec//readme.md)
3. **Download SD model** : download ["v1-5-pruned.ckpt"](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). And put it on the directory named "./models".
4. **Convert weight of sd model** : 
   ```python
   python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
   ```
5. **Config training setting** : [Look this](./config.py)
6. **Start to train a AdaBLDM**:
    ```python
    # default : train a hazelnut with hole.
    python train.py
    ```

## Test
0. **Input the model checkpoint** :  in [test.py line 35](./test.py).
1. **Run code**:
    ```python
    # default : test with hazelnut with hole.
    python test.py
    ```



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
## Acknowledgement
[ForegroundPredictor](https://github.com/flyinghu123/CPR)

[ControlNet](https://github.com/lllyasviel/ControlNet)

[LatentDiffusion](https://github.com/CompVis/latent-diffusion)

[StableDiffusion](https://github.com/CompVis/stable-diffusion)




## Citation
```shell
@article{Li2024ANA,
    title={A Novel Approach to Industrial Defect Generation through Blended Latent Diffusion Model with Online Adaptation},
    author={Hanxi Li and Zhengxun Zhang and Hao Chen and Lin Wu and Bo Li and Deyin Liu and Mingwen Wang},
    year={2024},
    url={https://api.semanticscholar.org/CorpusID:268091266},
}
```





