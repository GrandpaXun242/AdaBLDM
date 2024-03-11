#!/bin/bash
sourcedir=./log/mvtec_retreival_foreground_12_320_features.2_features.2_EfficientnetPM
targetdir=/your/path/mvtec_fewshot
categories=`ls ${sourcedir}`
modes=( 'train' 'test' )
for cate in ${categories}
do
foregound_dir=${targetdir}/${cate}/foregound
mkdir ${foregound_dir}
    for mode in ${modes[@]}
    do
    t=${sourcedir}/${cate}/${mode}
    cp -r ${t} ${foregound_dir}/${mode}
    done

done