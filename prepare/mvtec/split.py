"""Select few shot samples from MVtec each category.
2023 06 14  13:56:19 CST
Setting : select number - 10
"""
import glob
import os
import random
from functools import partial
import shutil
from config import MVTEC_LIST, MVTECSOURCEROOT, SAVEPATH
from utils import get_dataset_categories

random.seed(40)

get_mvtecsub_path = partial(os.path.join, MVTECSOURCEROOT)
SELECTNUM = 10
SELECTRATE = 0.5
img_postfix = "png"


def split_init():
    for obj in MVTEC_LIST:
        with open(os.path.join(SAVEPATH, f"{obj}_VerboseInfo.txt"), "w") as f:
            print("Now start to generate the result of ", obj)
            mvtec_test_path = get_mvtecsub_path(obj, "test")
            mvtec_groundtruth_path = get_mvtecsub_path(obj, "ground_truth")
            categories = get_dataset_categories(mvtec_test_path)
            for cate in categories:
                select_save_path = os.path.join(SAVEPATH, obj, "fewshot", cate)
                remain_save_path = os.path.join(SAVEPATH, obj, "test", cate)
                for tmp in [select_save_path, remain_save_path]:
                    os.makedirs(tmp, exist_ok=True)
                img_list = glob.glob(
                    os.path.join(mvtec_test_path, cate, f"*.{img_postfix}")
                )
                random.shuffle(img_list)
                select_list = img_list[:SELECTNUM]
                remain_list = img_list[SELECTNUM:]
                f.write(f"Select imgs of {cate}:\n")
                for sl in select_list:
                    # copy select data
                    shutil.copy(
                        sl, os.path.join(select_save_path, os.path.basename(sl))
                    )
                    f.write(f"{os.path.basename(sl)}\n")
                for rl in remain_list:
                    # copy remain data
                    shutil.copy(
                        rl, os.path.join(remain_save_path, os.path.basename(rl))
                    )
            # copy ground truth
            shutil.copytree(
                mvtec_groundtruth_path, os.path.join(SAVEPATH, obj, "ground_truth")
            )
            shutil.copytree(
                get_mvtecsub_path(obj, "train"), os.path.join(SAVEPATH, obj, "train")
            )


def split_from_txt(task_dir: str):
    obj_list = glob.glob(os.path.join(task_dir, "*.txt"))
    for obj in obj_list:
        obj_name = os.path.basename(obj).rsplit("_", 1)[0]
        mvtec_groundtruth_path = get_mvtecsub_path(obj_name, "ground_truth")
        mvtec_test_path = get_mvtecsub_path(obj_name, "test")
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
        with open(os.path.join(SAVEPATH, f"{obj_name}_VerboseInfo.txt"), "w") as f:
            for cate in cates.keys():
                f.write(f"Select imgs of {cate}:\n")
                select_save_path = os.path.join(SAVEPATH, obj_name, "fewshot", cate)
                remain_save_path = os.path.join(SAVEPATH, obj_name, "test", cate)
                img_list = glob.glob(
                    os.path.join(mvtec_test_path, cate, f"*.{img_postfix}")
                )
                for tmp in [select_save_path, remain_save_path]:
                    os.makedirs(tmp, exist_ok=True)
                for sl in cates[cate]:
                    # copy select data
                    sl = os.path.join(mvtec_test_path, cate, sl)
                    shutil.copy(
                        sl, os.path.join(select_save_path, os.path.basename(sl))
                    )
                    f.write(f"{os.path.basename(sl)}\n")
                for rl in img_list:
                    if rl in cates[cate]:
                        continue
                    rl = os.path.join(mvtec_test_path, cate, rl)
                    # copy remain data
                    shutil.copy(
                        rl, os.path.join(remain_save_path, os.path.basename(rl))
                    )
        # copy ground truth
        shutil.copytree(
            mvtec_groundtruth_path, os.path.join(SAVEPATH, obj_name, "ground_truth")
        )
        shutil.copytree(
            get_mvtecsub_path(obj_name, "train"),
            os.path.join(SAVEPATH, obj_name, "train")
        )


if __name__ == "__main__":
    split_from_txt("./split_task")
