save_memory = False

#TODO change by yourself
# FEWSHOTDIR same with the  value of "FEWSHOTSAVEPATH" which is setted in "AdaBLDM/prepare/mvtec/config.py"
FEWSHOTDIR = "/home/zzx/repository/DataSet/mvtec_fewshot"

# --------------------#
# Dataset Config
# --------------------#
# MVTec
# hazelnut hole

#TODO change by yourself
# hazelnut_hole_binary_triple is selected from "augment_contorl_data_save_path" which is setted in "AdaBLDM/prepare/mvtec/generate_dataset.py"
hazelnut_hole_binary_triple = (
    "./ControlNet_repository/triple_binary/hazelnut/hole"
)

DATASETROOT = hazelnut_hole_binary_triple
