"""
2023年 06月 15日 星期四 14:08:05 CST
Generate fusion img
"""
from share import *
import glob
import config
import matplotlib.pyplot as plt

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import os
import shutil
from config import FEWSHOTDIR

from scipy.ndimage import binary_dilation
import torch.nn.functional as F
import tqdm

import copy

model = create_model('./models/cldm_v15.yaml').cpu()
# TODO change the weight path in line 35
model.load_state_dict(
    load_state_dict(
        "lightning_logs/version_41/checkpoints/epoch=9-step=29999.ckpt",
        location='cpu',
    )
)

model = model.cuda()
ddim_sampler = DDIMSampler(model)


def tensor2img(tensor: torch.Tensor):
    a = (
        (einops.rearrange(tensor, 'b c h w -> b h w c') * 127.5 + 127.5)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )
    return a


def np2tensor(source: np.ndarray, reverse: bool = True):
    source = torch.from_numpy(source.copy()).float().cuda() / 255.0
    source = source.unsqueeze(1)
    if reverse:
        source = 1.0 - source
    return source


def checkLTPoint(empty: np.ndarray, anoamly: np.ndarray):
    H, W = empty.shape
    Ha, Wa = anoamly.shape
    Hr = H - Ha
    Wr = W - Wa
    h = random.choice(range(Hr))
    w = random.choice(range(Wr))
    return h, w


def generate_triple_map(
    source_img: np.ndarray,
    test_fn,
    foreground_prefix=None,
    obj_type=None,
    cate=None,
    fg_size=(256, 256),
):
    """_summary_

    Args:
        source_img (np.ndarray): np
        test_fn (_type_): path of source img
    """
    foreground_prefix = os.path.join(FEWSHOTDIR, obj_type, "foregound", "train", "good")
    select_train_set = glob.glob(
        os.path.join(FEWSHOTDIR, obj_type, "fewshot", cate, "*.*g")
    )
    select_train_set = [os.path.basename(i) for i in select_train_set]
    mask_prefix = os.path.join(FEWSHOTDIR, obj_type, "ground_truth", cate)
    masks = [os.path.join(mask_prefix, i[:-4] + "_mask.png") for i in select_train_set]
    basename = os.path.basename(test_fn)[:-4]
    fg_fn = os.path.join(foreground_prefix, f"f_{basename}.npy")
    if os.path.exists(fg_fn):
        # Object's foreground prediction
        n = np.load(fg_fn)
        n = n * 255
        n = n.astype(np.uint8)
        n = cv2.resize(n, dsize=fg_size)
        kernel = np.ones((5, 5), np.uint8)
        T, img = cv2.threshold(n, 127, 255, cv2.THRESH_OTSU)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    else:
        # Texture's foreground prediction
        img = np.ones(fg_size, np.uint8) * 255
    mask_np = cv2.imread(random.choice(masks), 0)
    mask_np = cv2.resize(mask_np, dsize=fg_size)
    m = mask_np
    y, x = np.where(m > 0)
    y0 = y.min()
    x0 = x.min()
    y1 = y.max()
    x1 = x.max()
    crop_anomaly = m[y0:y1, x0:x1]
    m_sum = crop_anomaly > 0
    m_sum = m_sum.sum()

    def generate_map(img, crop_anomaly, m_sum):
        """obtain triple map by confusing object mask and defect mask.

        Args:
            img (_type_): source img
            crop_anomaly (_type_): _description_
            m_sum (_type_): _description_

        Returns:
            np.ndarray: triple map
        """
        place_h, place_w = checkLTPoint(img, crop_anomaly)
        empty_mask = np.zeros_like(img)
        empty_mask[
            place_h : place_h + crop_anomaly.shape[0],
            place_w : place_w + crop_anomaly.shape[1],
        ] = crop_anomaly
        t1 = empty_mask > 0
        t2 = img > 0
        fusion_sum = t1 * t2
        fusion_sum = fusion_sum > 0
        fusion_sum = fusion_sum.sum()
        if fusion_sum == m_sum:
            object_mask = img[..., None]
            object_mask = np.concatenate([object_mask] * 3, axis=2)
            object_mask[object_mask > 0] = 127
            empty_mask = empty_mask[..., None]
            empty_mask[empty_mask > 0] = 255
            empty_mask = np.concatenate([empty_mask] * 3, axis=2)
            object_mask[empty_mask > 0] = 0
            object_mask += empty_mask
            if False:  # visualizztion
                plt.subplot(221)
                plt.imshow(source_img)
                plt.subplot(222)
                plt.imshow(img)
                plt.subplot(223)
                plt.imshow(empty_mask)
                plt.subplot(224)
                plt.imshow(object_mask)
                plt.show()
            return object_mask
        else:
            object_mask = generate_map(img, crop_anomaly, m_sum)
            return object_mask

    object_mask = generate_map(img, crop_anomaly, m_sum)
    return object_mask


# fmt:off
def test_by_triple_map(
    iteration              = 1,
    image_resolution       = 256,
    num_samples            = 1,
    seed                   = 102,
    guess_mode             = False,
    ddim_steps             = 10,
    strength               = 1,
    scale                  = 9.0,
    eta                    = 0.0,
    a_prompt               = '',
    n_prompt               = '',
    pseudo_image_save_path = "",
    obj_type               = "hazelnut",
    cate                   = "print",
    pixel_image_size       = (256, 256),
    use_dilation           = True,
    save_result            = False,
):
    #fmt:on
    prompt = f"a {obj_type} with a {cate},black background."
    if len(pseudo_image_save_path) == 0:
        assert (
            "Please input the pseudo_image_save_path , pseudo iamge save path is empty."
        )
    # fm:off
    good_train_path_prefix = f"{FEWSHOTDIR}/{obj_type}/train/good"
    good_list = glob.glob(os.path.join(good_train_path_prefix, "*.*g"))
    good_candidate = random.choices(good_list, k=num_samples)
    source_list = []
    mask_list = []
    # fm:on
    try:
        for i in range(num_samples):
            source_image = cv2.imread(good_candidate[i])
            source_image = cv2.resize(source_image, pixel_image_size)
            source_image = source_image[..., ::-1]
            mask_image = generate_triple_map(
                source_image, good_candidate[i], obj_type=obj_type, cate=cate
            )
            source_list.append(source_image)
            mask_list.append(mask_image)
    except Exception as e:
        print(e)
        print("Exception , triple map.")
        return
    mask_list = np.array(mask_list)
    source_list = np.array(source_list)
    pt_masks_show = mask_list.copy()  # Pixel triple Mask
    ctn_input_image = mask_list  # Control net input
    ctn_input_image = ctn_input_image.astype(np.uint8)

    detected_maps = []
    latent_masks = []
    origin_masks = []
    for i in range(num_samples):
        img = resize_image(HWC3(ctn_input_image[i]), image_resolution)
        H, W, C = img.shape
        detected_map = HWC3(img)
        detected_maps.append(detected_map)
        m = cv2.resize(img[..., 0], (32, 32), interpolation=cv2.INTER_NEAREST)
        m[m < 200] = 0
        if use_dilation:  # Using dilation mask in latent space.
            k_size = 3 + 2 * 1
            m = binary_dilation(m, structure=np.ones((k_size, k_size)))
            temm = np.zeros_like(m, dtype=np.uint8)
            temm[m] = 255
            latent_masks.append(temm)
        else:
            latent_masks.append(m)
        org = cv2.resize(img[..., 0], pixel_image_size)
        org[org < 200] = 0
        org[org > 0] = 255
        origin_masks.append(org)

    latent_masks = np.array(latent_masks)
    origin_masks = np.array(origin_masks)

    latent_mask_clone = latent_masks.copy()
    latent_masks = np2tensor(latent_masks.copy())
    origin_masks = np2tensor(origin_masks.copy())  # pixel image mask

    detected_maps = np.array(detected_maps)
    control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    x0 = torch.from_numpy(source_list.copy()).float().cuda() / 255.0
    x0 = torch.einsum("bhwc->bchw", x0)
    x0 = (x0 - 0.5) / 0.5
    encoder_posterior = model.encode_first_stage(x0)
    z = model.get_first_stage_encoding(encoder_posterior).detach()
    m = latent_masks

    if seed == -1:
        seed = random.randint(0, 65535)

    # fmt: off
    cond = { "c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)], "empty_crossattn":[model.get_learned_conditioning([""] * num_samples)]}
    un_cond = { "c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], "empty_crossattn":[model.get_learned_conditioning([""] * num_samples)]}
    shape = (4, H // 8, W // 8)
    model.control_scales = ( [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13))  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    # normal geneartion
    if False:
        samples, intermediates = ddim_sampler.sample( ddim_steps, num_samples, shape, cond, x0=z, mask=m, verbose=False, eta=eta, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond,)
        x_samples = model.decode_first_stage(samples)
        x_samples = tensor2img(x_samples)
        # without mask
        samples_defect, _ = ddim_sampler.sample( ddim_steps, num_samples, shape, cond, verbose=False, eta=eta, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond,)
        without_patse = samples_defect.clone()
        without_patse = model.decode_first_stage(without_patse)
        without_patse = tensor2img(without_patse)

        samples_defect = model.decode_first_stage(samples_defect)
        samples_defect = tensor2img(samples_defect)

    # blended
    cutoff_index = 45
    blended, blended_intermeid = ddim_sampler.sample_blended( ddim_steps, num_samples, shape, cond, mask=m,verbose=False, eta=eta, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond, org_mask=origin_masks , init_image=x0,cutoff_index=cutoff_index) 

    curr_latent = blended.clone().detach()

    blended  = model.decode_first_stage(blended)
    fg_image = blended.clone().detach()

    blended  = tensor2img(blended)
    bg_image = x0.clone().detach()
    bg_mask  = origin_masks.clone().detach()

    decoder_copy = copy.deepcopy(model.first_stage_model.decoder)
    model.first_stage_model.decoder.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.first_stage_model.decoder.parameters(), lr=0.0001)
    # opt_schedule = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.1)

    def recon_loss( fg_image: torch.Tensor, bg_image: torch.Tensor, curr_latent: torch.Tensor, mask: torch.Tensor, preservation_ratio: float = 100, model: torch.nn.Module = None,):
        curr_latent = 1.0 / model.scale_factor * curr_latent
        curr_reconstruction:torch.Tensor = model.first_stage_model.decode(curr_latent)
        loss = ( F.mse_loss(fg_image * mask, curr_reconstruction * mask) + F.mse_loss(bg_image * (1 - mask), curr_reconstruction * (1 - mask)) * preservation_ratio)
        return loss

    #TODO Finetune Phase
    for i in tqdm.tqdm(range(200)):
        optimizer.zero_grad()
        loss = recon_loss( fg_image=fg_image, bg_image=bg_image, curr_latent=curr_latent, mask=1 - bg_mask, model=model)
        loss.backward()
        optimizer.step()
        # opt_schedule.step()

    weight_opt = model.decode_first_stage(curr_latent)
    weight_opt = tensor2img(weight_opt)

    reconstruction = model.decode_first_stage(z)
    reconstruction = tensor2img(reconstruction)
    model.first_stage_model.decoder = None
    model.first_stage_model.decoder = decoder_copy

    # fmt:on

    if False:  # Show the intermedia image
        for e, i in enumerate(blended_intermeid['x_inter']):
            i = model.decode_first_stage(i)
            i = tensor2img(i)
            cv2.imwrite(f"./intermeid/{e}.png", i[0][..., ::-1])

    for i in range(num_samples):
        if save_result:
            # Obtain binary map as gt.
            pb_masks_show = pt_masks_show[i].copy()
            pb_masks_show[pb_masks_show < 200] = 0
            cv2.imwrite(
                f"{pseudo_image_save_path}/mask/{iteration}_{i:03d}.png", pb_masks_show
            )
            cv2.imwrite(
                f"{pseudo_image_save_path}/source/{iteration}_{i:03d}.png",
                source_list[i][..., ::-1],
            )
            cv2.imwrite(
                f"{pseudo_image_save_path}/blended/{iteration}_{i:03d}.png",
                blended[i][..., ::-1],
            )
            cv2.imwrite(
                f"{pseudo_image_save_path}/bg_refine/{iteration}_{i:03d}.png",
                weight_opt[i][..., ::-1],
            )
        else:  # show_image
            plt.subplot(221)
            plt.title("reconstruction of good image")
            plt.imshow(reconstruction[i])
            plt.subplot(222)
            plt.title("latent mask")
            plt.imshow(latent_mask_clone[i])
            plt.subplot(223)
            plt.title("blended")
            plt.imshow(blended[i])
            plt.subplot(224)
            plt.title("refine image")
            plt.imshow(weight_opt[i])
            if True:
                pb_masks_show = pt_masks_show[i].copy()
                pb_masks_show[pb_masks_show < 200] = 0
                cv2.imwrite(f"./temp_any/mask.png", pb_masks_show)
                cv2.imwrite("./temp_any/source.png", source_list[i][..., ::-1])
                cv2.imwrite("./temp_any/blended.png", blended[i][..., ::-1])
                cv2.imwrite("./temp_any/weight_opt.png", weight_opt[i][..., ::-1])
                cv2.imwrite("./temp_any/reconstruct.png", reconstruction[i][..., ::-1])
            plt.show()


if __name__ == "__main__":
    if True:  # Generate final image based on triple mapi
        obj_name = "hazelnut"
        cate = "hole"
        ss = f"{FEWSHOTDIR}/{obj_name}/pesudo_result/{cate}"
        sss = [
            # "latentpaste",
            "mask",
            "stepbysteppaste",
            # "withoutpatse",
            "source",
            "blended",
            "bg_refine",
        ]
        for ii in sss:
            temp_path = os.path.join(ss, ii)
            shutil.rmtree(temp_path, ignore_errors=True)
            os.makedirs(temp_path, exist_ok=True)
        for i in range(625):
            test_by_triple_map(
                iteration=i,
                pseudo_image_save_path=ss,
                obj_type=obj_name,
                cate=cate,
                num_samples=1,
                ddim_steps=50,
                save_result=True
            )
