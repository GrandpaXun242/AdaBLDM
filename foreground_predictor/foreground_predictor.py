import sys
sys.path.append('./')
from glob import glob
from loguru import logger
from tqdm import tqdm
import cv2 as cv
import json
import numpy as np
import os
from models import  EfficientnetPM
import pickle
import torch
from futils import DATASET_INFOS, read_image, transform_img, ForegroundPredictor


dataset_name = 'mvtec'
resize = 320
device = 'cuda:0'
model_class = EfficientnetPM
# foreground
foreground_layer = 'features.2'
# retrieval
retrieval_layer = 'features.2'
retrieval_n_clusters = 12
knn = 10

# commom
CLASS_NAMES, object_classnames, texture_classnames = DATASET_INFOS[dataset_name]
data_root = f'./data/{dataset_name}'
assert os.path.exists(data_root) == True,f"{data_root} is not exists."
os.makedirs("./log" , exist_ok=True)
output_dir = f'./log/{dataset_name}_retreival_foreground_{retrieval_n_clusters}_{resize}_{foreground_layer}_{retrieval_layer}_{model_class.__name__}'
layers = list(set([foreground_layer, retrieval_layer]))
model = model_class(layers=layers)
model.to(device)
model.eval()
for classname in CLASS_NAMES:
    logger.info(classname)
    cur_classname_output_dir = os.path.join(output_dir, classname)
    os.makedirs(cur_classname_output_dir, exist_ok=True)
    train_features = {}
    train_image = {}
    for fn in tqdm(sorted(glob(os.path.join(data_root, classname, 'train/*/*'))), desc='extract train features', leave=False):
        k = os.path.relpath(fn, os.path.join(data_root, classname))
        image = read_image(fn, (resize, resize))
        image_t = transform_img(image)
        features_d = model.get_features(image_t[None].to(device))
        train_features[k] = features_d
        train_image[k] = image
    if classname in object_classnames:
        foreground_features = torch.cat([v[foreground_layer] for v in train_features.values()])  # b x 512 x h x w
        logger.info('foreground')
        foreground_predictor = ForegroundPredictor(device)
        foreground_predictor.fit(foreground_features)
        del foreground_features
        for fn in tqdm(sorted(glob(os.path.join(data_root, classname, 'train/*/*'))) + sorted(glob(os.path.join(data_root, classname, 'test/*/*'))), desc='predict data', leave=False):
            k = os.path.relpath(fn, os.path.join(data_root, classname))
            image = read_image(fn, (resize, resize))
            image_t = transform_img(image)
            features_d = model.get_features(image_t[None].to(device))
            features = features_d[foreground_layer]  # b x 512 x h x w
            lda_predict_norm = foreground_predictor.transform(features)
            lda_predict_norm = lda_predict_norm.cpu().numpy()
            # 保存
            cur_save_dir = os.path.dirname(os.path.join(cur_classname_output_dir, k))
            os.makedirs(cur_save_dir, exist_ok=True)
            cur_image_name = os.path.basename(k).split('.', 1)[0]
            cv.imwrite(os.path.join(cur_save_dir, f'f_{cur_image_name}.png'), lda_predict_norm*255.)
            np.save(os.path.join(cur_save_dir, f'f_{cur_image_name}.npy'), lda_predict_norm)
        with open(os.path.join(cur_classname_output_dir, f'foreground_predictor.pkl'), 'wb') as f:
            pickle.dump(foreground_predictor, f)