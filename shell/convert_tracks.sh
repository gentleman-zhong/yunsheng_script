#!/bin/bash

# 获取当前脚本所在路径
export PYTHONPATH="$PYTHONPATH:/code/SuperBuild/install/lib/python3.9/dist-packages:/code/SuperBuild/install/lib/python3.8/dist-packages:/code/SuperBuild/install/bin/opensfm"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/code/SuperBuild/install/lib"

BASE_PATH="/home/zhangzhong/experiment/one_km/099-4-7地块-routeid-76386/submodels"

for i in $(seq -w 37 37); do
    # 将 i 转换成四位数格式
    i_formatted=$(printf "%04d" $i)
    DATA_PATH="$BASE_PATH/submodel_$i_formatted/opensfm"
    
    # 检查文件夹是否存在
    if [ -d "$DATA_PATH" ]; then
        echo "Processing $DATA_PATH ..."
        python3 /home/zhangzhong/experiment/tianfugongyu/script/convert_tracks_format.py --data_path "$DATA_PATH"
    else
        echo "Skipped $DATA_PATH, folder does not exist."
    fi
done

# python3 /home/zhangzhong/experiment/tianfugongyu/script/convert_tracks_format.py --data_path /home/zhangzhong/experiment/six_datasets_shanhaijing/250718_dongman/ori_cc_xml/dongmanAT/ --to_binary
# python3 /home/zhangzhong/experiment/tianfugongyu/script/convert_tracks_format.py --data_path /home/zhangzhong/experiment/six_datasets_shanhaijing/250718_gongsi/ori_cc_xml/gongsiAT/ --to_binary
# python3 /home/zhangzhong/experiment/tianfugongyu/script/convert_tracks_format.py --data_path /home/zhangzhong/experiment/six_datasets_shanhaijing/250718_guanyin/ori_cc_xml/guanyinAT/ --to_binary
# python3 /home/zhangzhong/experiment/tianfugongyu/script/convert_tracks_format.py --data_path /home/zhangzhong/experiment/six_datasets_shanhaijing/250718_houzi/ori_cc_xml/houziAT/ --to_binary
# python3 /home/zhangzhong/experiment/tianfugongyu/script/convert_tracks_format.py --data_path /home/zhangzhong/experiment/six_datasets_shanhaijing/250718_tianfu/ori_cc_xml/tianfugongyuAT/ --to_binary