# BTAD
0. Change config.py
1. Run code to select the samples (only 10)  used in `A Novel Approach to Industrial Defect Generation through Blended Latent Diffusion Model with Online Adaptation`
    ``` shell
    python split.py 
    ```
2. Change the saving trainset path  in [generate_dataset.py](./generate_dataset.py) 237 line. (**This trainset is used for training ADABLDM.**)

3. Run code to generate trainset.
    ``` shell
    python generate_dataset.py
    ```
