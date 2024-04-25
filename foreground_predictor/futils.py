from loguru import logger
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2 as cv
import numpy as np
import patchcore.backbones
import patchcore.patchcore
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
from typing import List, Tuple, Dict, Sequence

DATASET_INFOS = {
    'mvtec': [
        [
            'bottle',
            'cable',
            'capsule',
            'carpet',
            'grid',
            'hazelnut',
            'leather',
            'metal_nut',
            'pill',
            'screw',
            'tile',
            'toothbrush',
            'transistor',
            'wood',
            'zipper',
        ],
        [
            'bottle',
            'cable',
            'capsule',
            'hazelnut',
            'metal_nut',
            'pill',
            'screw',
            'toothbrush',
            'transistor',
            'zipper',
        ],
        ['carpet', 'grid', 'leather', 'tile', 'wood'],
    ],  # all, obj, texture
    'btad': [["01", "02", "03"], ["01", "03"], ["02"]],
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



def read_image(path, resize=None):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if resize:
        img = cv.resize(img, dsize=resize)
    return img

transform_img = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)



class FMinMaxScaler:
    def __init__(self, ratio=0.01):
        self.ratio = ratio
        self.min = None
        self.max = None

    def fit(self, data):
        m0 = np.partition(data, int(data.shape[0] * self.ratio), axis=0)[
            int(data.shape[0] * self.ratio) - 1
        ]
        m1 = np.partition(data, -int(data.shape[0] * self.ratio), axis=0)[
            -int(data.shape[0] * self.ratio)
        ]
        data = data[(data >= m0) & (data <= m1)]
        self.min = data.min(0).item()
        self.max = data.max(0).item()

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    @torch.no_grad()
    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = np.clip(data, self.min, self.max)
        elif isinstance(data, torch.Tensor):
            data = torch.clamp(data, self.min, self.max)
        data = (data - self.min) / (self.max - self.min)
        return data

class LinearDiscriminantAnalysis_Pytorch:

    def __init__(self, lda: LinearDiscriminantAnalysis, device='cuda') -> None:
        self.coef = torch.from_numpy(lda.coef_.T).to(device)
        self.intercept = torch.from_numpy(lda.intercept_).to(device)

    @torch.no_grad()
    def transform(self, x):
        return x @ self.coef + self.intercept

class ForegroundPredictor:
    gaussian_filter_sigma_ratio = 1 / 40

    def __init__(self, device='cuda', seed=66, n_clusters=2) -> None:
        self.device = device
        self.kmeans_f_num = 50000
        self.lda_f_num = 15000
        self.foreground_ratio = 0.2
        self.background_ratio = -3
        self.n_clusters = n_clusters
        self.random_state = np.random.RandomState(seed)
        self.lda = LinearDiscriminantAnalysis()
        self.kmeans = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.lda_predictor = None
        self.normalizer = FMinMaxScaler()

    @classmethod
    def g(cls, lda_predict_norm):
        return lda_predict_norm

    def fit(self, foreground_features):
        B, C, H, W = foreground_features.shape
        image_foreground_features = (
            foreground_features.permute(0, 2, 3, 1).cpu().numpy()
        )  # b x h x w x 512
        self.kmeans.fit(
            image_foreground_features.reshape(-1, foreground_features.shape[1])[
                self.random_state.permutation(B * H * W)[: self.kmeans_f_num]
            ]
        )
        labels_imgs = self.kmeans.predict(
            image_foreground_features.reshape(-1, foreground_features.shape[1])
        ).reshape(B, H, W)
        if self.background_ratio < 0:
            self.background_ratio = -self.background_ratio / labels_imgs.shape[1]
        background_mask = np.zeros((B, H, W), dtype=bool)
        background_mask[
            :, : int(self.background_ratio * labels_imgs.shape[1]), :
        ] = True
        background_mask[
            :, -int(self.background_ratio * labels_imgs.shape[1]) :, :
        ] = True
        background_mask[
            :,
            int(self.background_ratio * labels_imgs.shape[1]) : -int(
                self.background_ratio * labels_imgs.shape[1]
            ),
            : int(self.background_ratio * labels_imgs.shape[2]),
        ] = True
        background_mask[
            :,
            int(self.background_ratio * labels_imgs.shape[1]) : -int(
                self.background_ratio * labels_imgs.shape[1]
            ),
            -int(self.background_ratio * labels_imgs.shape[2]) :,
        ] = True
        bidx, hidx, widx = np.where(background_mask)
        background_features = image_foreground_features[bidx, hidx, widx, :]
        background_labels = labels_imgs[bidx, hidx, widx]
        one_hot = np.zeros((background_labels.shape[0], self.kmeans.n_clusters))
        one_hot[np.arange(one_hot.shape[0]), background_labels] = 1
        hist = one_hot.sum(0)
        background_label_u = [hist.argmax()]
        background_p_mask = (
            np.stack([background_labels == l for l in background_label_u], 1).sum(1) > 0
        )
        background_features = background_features[background_p_mask]
        background_label = np.zeros((background_features.shape[0]), dtype=int)

        foreground_mask = np.zeros((B, H, W), dtype=bool)
        foreground_mask[
            :,
            int(
                labels_imgs.shape[1] / 2 - labels_imgs.shape[1] * self.foreground_ratio
            ) : int(
                labels_imgs.shape[1] / 2 + labels_imgs.shape[1] * self.foreground_ratio
            ),
            int(
                labels_imgs.shape[2] / 2 - labels_imgs.shape[2] * self.foreground_ratio
            ) : int(
                labels_imgs.shape[2] / 2 + labels_imgs.shape[2] * self.foreground_ratio
            ),
        ] = True

        bidx, hidx, widx = np.where(foreground_mask)
        foreground_features = image_foreground_features[bidx, hidx, widx, :]
        foreground_labels = labels_imgs[bidx, hidx, widx]
        foreground_p_mask = np.stack(
            [foreground_labels != l for l in background_label_u], 1
        ).sum(1) >= len(background_label_u)
        foreground_features = foreground_features[foreground_p_mask]
        foreground_label = np.ones((foreground_features.shape[0]), dtype=int)
        background_idx = self.random_state.permutation(len(background_features))[
            : self.lda_f_num
        ]
        foreground_idx = self.random_state.permutation(len(foreground_features))[
            : self.lda_f_num
        ]
        background_features = background_features[background_idx]
        foreground_features = foreground_features[foreground_idx]
        background_label = background_label[background_idx]
        foreground_label = foreground_label[foreground_idx]
        self.lda.fit_transform(
            np.concatenate([background_features, foreground_features]),
            np.concatenate([background_label, foreground_label]),
        )
        self.normalizer.fit(
            self.lda.decision_function(
                image_foreground_features.reshape(
                    -1, image_foreground_features.shape[-1]
                )
            )
        )
        self.lda_predictor = LinearDiscriminantAnalysis_Pytorch(self.lda, self.device)

    def transform(self, features):
        # features: 1 x c x h x w
        return self.normalizer.transform(
            self.lda_predictor.transform(
                features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            )
        ).reshape(features.shape[2], features.shape[3])

class LastLayerToExtractReachedException(Exception):
    pass
class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, stop_length: int = -1):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.stop_length = stop_length

    @torch.no_grad()
    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output.detach()
        if self.stop_length > 0 and len(self.hook_dict) >= self.stop_length:
            raise LastLayerToExtractReachedException()

class FeaturesCollector:
    def __init__(
        self,
        backbone: nn.Module,
        layers: List[str] = ['layer2'],
        interrupt: bool = True,
    ) -> None:
        self.backbone = backbone
        self.interrupt = interrupt
        self._features_d = {}
        self.removable_handles = []
        for layer in layers:
            forward_hook = ForwardHook(
                self._features_d, layer, -1 if not self.interrupt else len(layers)
            )
            network_layer = backbone
            while "." in layer:
                extract_block, layer = layer.split(".", 1)
                network_layer = network_layer.__dict__["_modules"][extract_block]
            network_layer = network_layer.__dict__["_modules"][layer]
            if isinstance(network_layer, torch.nn.Sequential):
                self.removable_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            elif isinstance(network_layer, torch.nn.Module):
                self.removable_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )

    @torch.no_grad()
    def __call__(self, x):
        try:
            self.backbone(x)
        except LastLayerToExtractReachedException:
            pass
        try:
            return self._features_d.copy()
        finally:
            self._features_d.clear()

    def __del__(self):
        for handle in self.removable_handles:
            handle.remove()