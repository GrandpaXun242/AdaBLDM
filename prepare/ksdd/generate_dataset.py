import cv2
import glob
import os
import matplotlib.pyplot as plt
import random
import shutil
import json
import numpy as np
from math import *
import tqdm
from config import KSDD_ROOT_PATH
random.seed(1003)
GENERATENUM = 20000 # previously 6000



def checkLTPoint(empty: np.ndarray, anoamly: np.ndarray):
    H, W, C = empty.shape
    Ha, Wa, C = anoamly.shape
    Hr = H - Ha
    Wr = W - Wa
    h = random.choice(range(Hr))
    w = random.choice(range(Wr))
    return h, w

def warpAnomaly(img, mask, useRotate=True, useResize=True):
    y, x, _ = np.where(mask > 0)
    y0, x0, y1, x1 = y.min(), x.min(), y.max(), x.max()
    img_result = img[y0:y1, x0:x1]
    mask_result = mask[y0:y1, x0:x1]
    height, width = img_result.shape[:2]

    if useRotate:
        x = random.randint(0, 360)
        degree = x
        M = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        heightNew = int(
            width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree)))
        )
        widthNew = int(
            height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree)))
        )

        M[0, 2] += (widthNew - width) / 2
        M[1, 2] += (heightNew - height) / 2
        img_result = cv2.warpAffine(
            img_result, M, (widthNew, heightNew), borderValue=(0, 0, 0)
        )
        mask_result = cv2.warpAffine(
            mask_result, M, (widthNew, heightNew), borderValue=(0, 0, 0)
        )
    if useResize:
        x = random.randint(-150, 150)
        x = x / 1000.0 + 1
        H_R = int(height * x)
        W_R = int(width * x)
        img_result = cv2.resize(img_result, dsize=(H_R, W_R))
        mask_result = cv2.resize(mask_result, dsize=(H_R, W_R))
    y, x, _ = np.where(mask_result > 0)
    y0, x0, y1, x1 = y.min(), x.min(), y.max(), x.max()
    mask_result[mask_result < 200] = 0
    mask_result[mask_result > 0] = 255
    img_result, mask_result = img_result[y0:y1, x0:x1], mask_result[y0:y1, x0:x1]
    img_result = img_result * (mask_result > 0)
    return img_result, mask_result


def make_data(
    dataRoot: str = KSDD_ROOT_PATH,
    obj="",
    save_path="",
    mask_path="",
    fore_mode="train",
    ishow=True,
    start_num=0,
    add_binary_map=False,
):
    annotaion_path = f"{save_path}/prompt.json"
    real_anomaly = os.path.join(dataRoot, "train", "ng")
    mask_anomaly = os.path.join(dataRoot,"train","gt")
    good = os.path.join(dataRoot,  "train", "good")
    # good_foreground_suffix = os.path.join(dataRoot, obj, "foregound", fore_mode, "good")
    good_foreground_suffix = ""
    print(real_anomaly)
    real_anomaly_list = glob.glob(os.path.join(real_anomaly, "*.png"))
    real_good_list = glob.glob(os.path.join(good, "*.png"))
    CNDsource_path = f"{save_path}/target"
    CNDtarget_path = f"{save_path}/source"

    # shutil.rmtree([CNDtarget_path, CNDsource_path], ignore_errors=True)
    os.makedirs(CNDsource_path, exist_ok=True)
    os.makedirs(CNDtarget_path, exist_ok=True)

    gene_num = start_num
    if gene_num >= GENERATENUM:
        return
    with open(annotaion_path, "w") as f:
        for index in range(start_num):
            f.write(
                f'{{"source":"source/{gene_num}_triple.png" , "target":"target/{gene_num}_defect.png","prompt":"a {obj} with a {cate}"}}\n'
            )
        while True:
            for index in range( start_num,GENERATENUM):
                # obtain the image path info
                a_item = real_anomaly_list[index % len(real_anomaly_list)]
                good_item = real_good_list[index % len(real_good_list)]
                m_item = os.path.join(
                    mask_anomaly, os.path.basename(a_item)
                )
                good_foreground = os.path.join(
                    good_foreground_suffix, f"f_{os.path.basename(good_item)[:-4]}.npy"
                )
                good_fn = os.path.basename(good_item)[:-4]
                a = cv2.imread(a_item)  # anomaly image
                m = cv2.imread(m_item)  # anomaly mask
                g = cv2.imread(good_item)  # good image
                good_img = g.copy()  # be used to save good image
                empty = np.zeros_like(g)
                empty_mask = np.zeros_like(m)
                # extract the mask region
                try:
                    img_result, mask_result = warpAnomaly(
                        a, m, useResize=True, useRotate=True
                    )
                    crop_anomaly = img_result
                    crop_mask = mask_result
                    place_h, place_w = checkLTPoint(empty, crop_anomaly)
                except:
                    continue
                m_sum = crop_anomaly > 0
                m_sum = m_sum.sum()
                empty[
                    place_h : place_h + crop_anomaly.shape[0],
                    place_w : place_w + crop_anomaly.shape[1],
                ] = crop_anomaly
                if os.path.exists(good_foreground):  # object
                    n = np.load(good_foreground)
                    n = n * 255
                    n = n.astype(np.uint8)
                    n = cv2.resize(n, dsize=(empty.shape[0], empty.shape[1]))
                    kernel = np.ones((5, 5), np.uint8)
                    T, fro_img = cv2.threshold(n, 127, 255, cv2.THRESH_BINARY)
                    fro_img = cv2.morphologyEx(fro_img, cv2.MORPH_CLOSE, kernel)
                    fro_img = np.stack([fro_img] * 3, axis=2)
                else:  # texture
                    fro_img = (
                        np.ones((empty.shape[0], empty.shape[1], 3), np.uint8) * 255
                    )
                g[empty > 0] = 0
                fusion_sum = (empty > 0) * (fro_img > 0)
                fusion_sum = fusion_sum > 0
                fusion_sum = fusion_sum.sum()

                # is covered?
                if fusion_sum == m_sum:
                    g = g + empty
                    empty = g
                    empty_mask[
                        place_h : place_h + crop_anomaly.shape[0],
                        place_w : place_w + crop_anomaly.shape[1],
                    ] = (
                        crop_mask * 255
                    )
                    fro_img = fro_img.sum(axis=2)
                    empty_mask = empty_mask.sum(axis=2)
                    empty_mask[empty_mask > 0] = 255
                    fro_img[fro_img > 0] = 127
                    binary_map = fro_img.copy()  # store
                    fro_img[empty_mask > 0] = 0
                    fro_img += empty_mask
                    fro_img = np.stack([fro_img] * 3, axis=2)
                    empty_mask = np.stack([empty_mask] * 3, axis=2)
                    binary_map = np.stack([binary_map] * 3, axis=2)
                    triple_map = fro_img.copy()
                    if ishow:  # show image
                        plt.subplot(231)
                        plt.imshow(empty[..., ::-1])
                        plt.subplot(232)
                        plt.imshow(img_result)
                        plt.subplot(233)
                        plt.title("triple_map")
                        plt.imshow(triple_map)
                        plt.subplot(234)
                        plt.imshow(empty_mask)
                        plt.subplot(235)
                        plt.title("binary map")
                        plt.imshow(binary_map)
                        plt.subplot(236)
                        plt.title("normal image")
                        plt.imshow(good_img)
                        plt.show()
                        # exit()
                    else:
                        # saving the triple map
                        cv2.imwrite(
                            f"{os.path.join(CNDsource_path , str( gene_num ))}_defect.png",
                            empty,
                        )
                        cv2.imwrite(
                            f"{os.path.join(CNDtarget_path , str( gene_num ))}_triple.png",
                            fro_img,
                        )

                        gs = f"{os.path.join(CNDsource_path , str( good_fn ))}_good.png"
                        if os.path.exists(gs) == False:
                            if add_binary_map:
                                # saving the binary map
                                cv2.imwrite(
                                    f"{os.path.join(CNDsource_path , str( good_fn ))}_good.png",
                                    good_img,
                                )
                                cv2.imwrite(
                                    f"{os.path.join(CNDtarget_path , str( good_fn ))}_binary.png",
                                    binary_map,
                                )

                        if add_binary_map:  # saveing the binary map
                            f.write(
                                f'{{"source":"source/{good_fn}_binary.png" , "target":"target/{good_fn}_good.png","prompt":"a {obj},black background."}}\n'
                            )
                        f.write(
                            f'{{"source":"source/{gene_num}_triple.png" , "target":"target/{gene_num}_defect.png","prompt":"a defect named crack on metal surface"}}\n'
                        )
                    gene_num += 1
                    if gene_num >= GENERATENUM:
                        break
            if gene_num >= GENERATENUM:
                break

if __name__ == "__main__":
    save_path= "/your/path/ksvdd_triple_dataset"
    make_data(save_path=save_path,ishow=False)