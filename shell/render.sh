#!/bin/bash

LOGFILE="/home/zhangzhong/experiment/six_datasets_shanhaijing/render_log.txt"
if [ -f "$LOGFILE" ]; then
    > "$LOGFILE"
fi
echo "========== 全部渲染与评估开始：$(date) ==========" > $LOGFILE

declare -a datasets=(
    "250718_gongsi"
    "250718_tianfu"
    "250718_dongman"
    "250718_guanyin"
    "250718_houzi"
    "250718_feilou"
)

for name in "${datasets[@]}"; do
    echo -e "\n========== 开始处理数据集：$name ==========" | tee -a $LOGFILE

    OBJ_FOLDER="/home/zhangzhong/experiment/six_datasets_shanhaijing/${name}/odm_texturing/"
    XML_DIR="/home/zhangzhong/experiment/six_datasets_shanhaijing/${name}/opensfm/"
    RENDER_SCRIPT="/home/zhangzhong/experiment/tianfugongyu/script/render/render.py"
    EVAL_SCRIPT="/home/zhangzhong/experiment/tianfugongyu/script/render/evaluate.py"

    echo "---- 渲染阶段 ----" | tee -a $LOGFILE
    blenderproc run "$RENDER_SCRIPT" --obj_folder "$OBJ_FOLDER" --xml_dir "$XML_DIR" >> $LOGFILE 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ 渲染成功：$name" | tee -a $LOGFILE
    else
        echo "❌ 渲染失败：$name" | tee -a $LOGFILE
    fi

    echo "---- 评估阶段 ----" | tee -a $LOGFILE
    # python "$EVAL_SCRIPT" --input_folder "$OBJ_FOLDER" >> $LOGFILE 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ 评估成功：$name" | tee -a $LOGFILE
    else
        echo "❌ 评估失败：$name" | tee -a $LOGFILE
    fi

    echo -e "========== 完成数据集：$name ==========\n" | tee -a $LOGFILE
done

echo "========== 全部完成：$(date) ==========" | tee -a $LOGFILE