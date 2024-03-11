import glob
from config import KSDD_ROOT_PATH
import cv2
import os
import tqdm
def resize(img_block,size= (224,224) , isMask = False):
    img_block = cv2.resize(img_block ,size)
    if isMask:
        img_block[img_block>127] = 255
        img_block[img_block<=127] = 0
    return img_block

"""
Ksdd download
https://www.vicos.si/resources/kolektorsdd2/
"""
if __name__ == "__main__":
    train_path = os.path.join(KSDD_ROOT_PATH,"train")
    test_path = os.path.join(KSDD_ROOT_PATH,"test")
    save_path = os.path.join(os.path.dirname(KSDD_ROOT_PATH) , "ksvdd_changedsize")
    #-----------------------#
    # Train set
    #-----------------------#
    for i in tqdm.tqdm(glob.glob(os.path.join(train_path,"*.*g")), desc = "Train"):
        bn = os.path.basename(i)
        if "GT" in bn or "copy" in bn:
            continue
        gt_path = os.path.join(os.path.dirname(i) , f"{bn[:-4]}_GT.png")

        img = cv2.imread(i)
        gt = cv2.imread(gt_path)
        h,w,c = img.shape
        # img block
        first_part = img[:w]
        sec_part = img[w:2*w]
        final_part = img[h-w:]
        # gt block
        gt_first_part = gt[:w]
        gt_sec_part = gt[w:2*w]
        gt_final_part = gt[h-w:]
        # resize
        first_part = resize(first_part)
        sec_part = resize(sec_part)
        final_part = resize(final_part)
        gt_first_part = resize(gt_first_part, isMask=True)
        gt_sec_part = resize(gt_sec_part , isMask=True)
        gt_final_part = resize(gt_final_part , isMask= True)
        #------------------#
        #      Save
        #------------------#
        for tmp_d in ["good","ng","gt"]:
            os.makedirs(os.path.join(save_path,"train",tmp_d),exist_ok=True)
        if gt_first_part.sum() >0:
            cv2.imwrite(os.path.join(save_path,"train","ng",f"{bn[:-4]}_first.png"),first_part)
        else:
            cv2.imwrite(os.path.join(save_path,"train","good",f"{bn[:-4]}_first.png"),first_part)
        if gt_sec_part.sum() >0:
            cv2.imwrite(os.path.join(save_path,"train","ng",f"{bn[:-4]}_sec.png"),sec_part)
        else:
            cv2.imwrite(os.path.join(save_path,"train","good",f"{bn[:-4]}_sec.png"),sec_part)
        if gt_final_part.sum() > 0 :
            cv2.imwrite(os.path.join(save_path,"train","ng",f"{bn[:-4]}_final.png"),final_part)
        else:
            cv2.imwrite(os.path.join(save_path,"train","good",f"{bn[:-4]}_final.png"),final_part)
        cv2.imwrite(os.path.join(save_path,"train","gt",f"{bn[:-4]}_first.png"),gt_first_part)
        cv2.imwrite(os.path.join(save_path,"train","gt",f"{bn[:-4]}_sec.png"),gt_sec_part)
        cv2.imwrite(os.path.join(save_path,"train","gt",f"{bn[:-4]}_final.png"),gt_final_part)

    #-----------------------#
    # Test set
    #-----------------------#
    for i in tqdm.tqdm(glob.glob(os.path.join(test_path,"*.*g")), desc = "Test"):
        bn = os.path.basename(i)
        if "GT" in bn or "copy" in bn:
            continue
        gt_path = os.path.join(os.path.dirname(i) , f"{bn[:-4]}_GT.png")
        img = cv2.imread(i)
        gt = cv2.imread(gt_path)
        h,w,c = img.shape
        first_part = img[:w]
        sec_part = img[w:2*w]
        final_part = img[h-w:]
        gt_first_part = gt[:w]
        gt_sec_part = gt[w:2*w]
        gt_final_part = gt[h-w:]
        first_part = resize(first_part)
        sec_part = resize(sec_part)
        final_part = resize(final_part)
        gt_first_part = resize(gt_first_part, isMask=True)
        gt_sec_part = resize(gt_sec_part , isMask=True)
        gt_final_part = resize(gt_final_part , isMask= True)
        #------------------#
        #      Save
        #------------------#
        for tmp_d in ["good","ng","gt"]:
            os.makedirs(os.path.join(save_path,"test",tmp_d),exist_ok=True)
        if gt_first_part.sum() >0:
            cv2.imwrite(os.path.join(save_path,"test","ng",f"{bn[:-4]}_first.png"),first_part)
        else:
            cv2.imwrite(os.path.join(save_path,"test","good",f"{bn[:-4]}_first.png"),first_part)
        if gt_sec_part.sum() >0:
            cv2.imwrite(os.path.join(save_path,"test","ng",f"{bn[:-4]}_sec.png"),sec_part)
        else:
            cv2.imwrite(os.path.join(save_path,"test","good",f"{bn[:-4]}_sec.png"),sec_part)
        if gt_final_part.sum() > 0 :
            cv2.imwrite(os.path.join(save_path,"test","ng",f"{bn[:-4]}_final.png"),final_part)
        else:
            cv2.imwrite(os.path.join(save_path,"test","good",f"{bn[:-4]}_final.png"),final_part)
        cv2.imwrite(os.path.join(save_path,"test","gt",f"{bn[:-4]}_first.png"),gt_first_part)
        cv2.imwrite(os.path.join(save_path,"test","gt",f"{bn[:-4]}_sec.png"),gt_sec_part)
        cv2.imwrite(os.path.join(save_path,"test","gt",f"{bn[:-4]}_final.png"),gt_final_part)

