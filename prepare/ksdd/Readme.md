0. Download the dataset of [KSDD2](https://www.vicos.si/resources/kolektorsdd2/)
1. Config the ksdd path in [config code](./config.py)
2. Runing the code to crop image
    ```python
    """
    Result : The directory named "ksdd_changedsize".

    Example :
            1.png -> [ 1_first.png , 1_sec.png , 1_final.png ]
            gt_1.png -> [ gt_1_first.png , gt_1_sec.png , gt_1_final.png ]
    """
    python prepare_ksdd.py
    ```
3. Change the saving trainset path  in [generate_dataset.py](./generate_dataset.py) 237 line. (**This trainset is used for training ADABLDM.**)

4. Run code to generate trainset.
    ``` shell
    python generate_dataset.py
    ```
