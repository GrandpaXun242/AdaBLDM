"""Select few shot samples from btad each category.
2023年 10月 6日 
Setting : select number - 10
"""
import glob
import os
import random
from functools import partial
import shutil
from config import BTAD_ROOT,objs,FEW_SAVE_PATH , file_suffix
import cv2
random.seed(40)
get_btad_subpath = partial(os.path.join , BTAD_ROOT)
SELECTNUM = 10
def split_init():
    """
    >>>  init split BTAD script.
    """
    os.makedirs(FEW_SAVE_PATH, exist_ok=True)
    for obj in objs:
        with open(os.path.join(FEW_SAVE_PATH , f"{obj}_VerboseInfo.txt") , "w") as f:
            print("Now start to generate the result of ", obj)
            btad_test_path = get_btad_subpath(obj,"test")
            btad_groundtruth_path = get_btad_subpath(obj, "ground_truth")
            categories = ["ko"]
            for cate in categories:
                select_save_path = os.path.join(FEW_SAVE_PATH, obj, "fewshot", cate)
                remain_save_path = os.path.join(FEW_SAVE_PATH, obj, "test", cate)
                for tmp in [select_save_path, remain_save_path]:
                    os.makedirs(tmp, exist_ok=True)
                img_list = sorted(glob.glob(
                    os.path.join(btad_test_path, cate, f"*{file_suffix[obj]}")
                ))
                random.shuffle(img_list)
                select_list = img_list[:SELECTNUM]
                remain_list = img_list[SELECTNUM:]
                f.write(f"Select imgs of {cate}:\n")
                for sl in select_list:
                    # copy select data
                    if sl.endswith("bmp"):
                        raw_img = cv2.imread(sl)
                        cv2.imwrite(os.path.join(select_save_path , os.path.basename(sl)[:-4]+".png") ,raw_img)
                        f.write(f"{os.path.basename(sl)[:-4]}.png\n")
                    else:
                        shutil.copy(
                            sl, os.path.join(select_save_path, os.path.basename(sl))
                        )
                        f.write(f"{os.path.basename(sl)}\n")
                for rl in remain_list:
                    # copy remain data
                    if rl.endswith("bmp"):
                        raw_img = cv2.imread(rl)
                        cv2.imwrite(os.path.join(remain_save_path , os.path.basename(rl)[:-4]+".png") ,raw_img)
                    else:
                        shutil.copy(
                            rl, os.path.join(remain_save_path, os.path.basename(rl))
                            
                        )
            # copy ground truth
            if file_suffix[obj] == ".png":
                shutil.copytree(
                    btad_groundtruth_path, os.path.join(FEW_SAVE_PATH, obj, "ground_truth")
                )
                shutil.copytree(
                    get_btad_subpath(obj, "train"), os.path.join(FEW_SAVE_PATH, obj, "train")
                )
            else:
                #ground_truth
                if obj == "01":
                    gt_list = sorted(glob.glob(
                        os.path.join(BTAD_ROOT , obj,"ground_truth", "ko", f"*.png")
                    ))
                else:
                    gt_list = sorted(glob.glob(
                        os.path.join(BTAD_ROOT , obj,"ground_truth", "ko", f"*{file_suffix[obj]}")
                    ))
                os.makedirs(os.path.join(FEW_SAVE_PATH , obj,"ground_truth","ko" ) ,exist_ok=True)
                for i in gt_list:
                    raw_img = cv2.imread(i)
                    cv2.imwrite(os.path.join(FEW_SAVE_PATH , obj,"ground_truth","ko" , os.path.basename(i)[:-4]+".png") ,raw_img)
                # train
                train_list = sorted(glob.glob(
                    os.path.join(BTAD_ROOT,obj,"train", "ok", f"*{file_suffix[obj]}")
                ))
                os.makedirs(os.path.join(FEW_SAVE_PATH , obj,"train","ok" ) ,exist_ok=True)
                for i in train_list:
                    raw_img = cv2.imread(i)
                    cv2.imwrite(os.path.join(FEW_SAVE_PATH , obj ,"train","ok" , os.path.basename(i)[:-4]+".png") ,raw_img)

def split_from_txt(task_dir: str):
    obj_list = glob.glob(os.path.join(task_dir, "*.txt"))
    for obj in obj_list:
        obj_name = os.path.basename(obj).rsplit("_", 1)[0]
        btad_test_path = get_btad_subpath(obj_name,"test")
        btad_groundtruth_path = get_btad_subpath(obj_name, "ground_truth")
        content = ""
        cates = {}
        category = ""
        with open(obj, "r") as f:
            content = f.readlines()
        for line in content:
            line = line.strip()
            if line.endswith("png"):
                cates[category].append(line)
            elif line.endswith(":"):
                category = line.rsplit(" ", 1)[-1][:-1]
                cates[category] = []
        os.makedirs(FEW_SAVE_PATH , exist_ok=True)
        with open(os.path.join(FEW_SAVE_PATH, f"{obj_name}_VerboseInfo.txt"), "w") as f:
            for cate in cates.keys():
                f.write(f"Select imgs of {cate}:\n")
                select_save_path = os.path.join(FEW_SAVE_PATH, obj_name, "fewshot", cate)
                remain_save_path = os.path.join(FEW_SAVE_PATH, obj_name, "test", cate)
                img_list = glob.glob(
                    os.path.join(btad_test_path, cate, f"*{file_suffix[obj_name]}")
                )
                for tmp in [select_save_path, remain_save_path]:
                    os.makedirs(tmp, exist_ok=True)
                for sl in cates[cate]:
                    # copy select data
                    sl = os.path.join(BTAD_ROOT , obj_name ,"test",cate,sl[:-4]+file_suffix[obj_name] )
                    raw_img = cv2.imread(sl)
                    cv2.imwrite(os.path.join(select_save_path , os.path.basename(sl)[:-4]+".png") ,raw_img)
                    f.write(f"{os.path.basename(sl)[:-4]}.png\n")
                for rl in img_list:
                    if rl in cates[cate]:
                        continue
                    rl = os.path.join(BTAD_ROOT , obj_name ,"test",cate,rl[:-4]+file_suffix[obj_name] )
                    raw_img = cv2.imread(rl)
                    cv2.imwrite(os.path.join(remain_save_path , os.path.basename(rl)[:-4]+".png") ,raw_img)
        if file_suffix[obj_name] == ".png":
            shutil.copytree(
                btad_groundtruth_path, os.path.join(FEW_SAVE_PATH, obj_name, "ground_truth")
            )
            shutil.copytree(
                get_btad_subpath(obj_name, "train"), os.path.join(FEW_SAVE_PATH, obj_name, "train")
            )
        else:
            #ground_truth
            if obj_name == "01":
                gt_list = sorted(glob.glob(
                    os.path.join(BTAD_ROOT , obj_name,"ground_truth", "ko", f"*.png")
                ))
            else:
                gt_list = sorted(glob.glob(
                    os.path.join(BTAD_ROOT , obj_name,"ground_truth", "ko", f"*{file_suffix[obj_name]}")
                ))
            os.makedirs(os.path.join(FEW_SAVE_PATH , obj_name,"ground_truth","ko" ) ,exist_ok=True)
            for i in gt_list:
                raw_img = cv2.imread(i)
                cv2.imwrite(os.path.join(FEW_SAVE_PATH , obj_name,"ground_truth","ko" , os.path.basename(i)[:-4]+".png") ,raw_img)
            # train
            train_list = sorted(glob.glob(
                os.path.join(BTAD_ROOT,obj_name,"train", "ok", f"*{file_suffix[obj_name]}")
            ))
            os.makedirs(os.path.join(FEW_SAVE_PATH , obj_name,"train","ok" ) ,exist_ok=True)
            for i in train_list:
                raw_img = cv2.imread(i)
                cv2.imwrite(os.path.join(FEW_SAVE_PATH , obj_name ,"train","ok" , os.path.basename(i)[:-4]+".png") ,raw_img)


if __name__ =="__main__":

    # split_init()
    split_from_txt("./split_task")

