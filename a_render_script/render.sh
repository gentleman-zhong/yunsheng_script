#!/bin/bash
# blenderproc run /home/zhangzhong1/experiment/tianfugongyu/script/a_render_script/render/render.py --obj_folder /home/zhangzhong1/experiment/CC_AT_ODM_reconstruction/tianfu/new_odm_texturing/ --xml_dir /home/zhangzhong1/experiment/CC_AT_ODM_reconstruction/tianfu/
# python /home/zhangzhong1/experiment/tianfugongyu/script/a_render_script/render/evaluate.py --input_folder /home/zhangzhong1/experiment/CC_AT_ODM_reconstruction/tianfu/new_odm_texturing/
LOGFILE="/home/zhangzhong1/experiment/CC_MVS_ODM_Mesh_teturing/render_log.txt"
if [ -f "$LOGFILE" ]; then
    > "$LOGFILE"
fi
echo "========== 全部渲染与评估开始：$(date) ==========" > $LOGFILE

declare -a datasets=(
    "houzi"
    # "tianfu"
    "gongsi"
    # "gongsi_only_odm"
    # "dongman"
    # "guanyin"
)

for name in "${datasets[@]}"; do
    echo -e "\n========== 开始处理数据集：$name ==========" | tee -a $LOGFILE

    OBJ_FOLDER="/home/zhangzhong1/experiment/CC_MVS_ODM_Mesh_teturing/${name}/odm_texturing/"
    XML_DIR="/home/zhangzhong1/experiment/CC_MVS_ODM_Mesh_teturing/${name}/"
    RENDER_SCRIPT="/home/zhangzhong1/experiment/tianfugongyu/script/a_render_script/render/render.py"
    EVAL_SCRIPT="/home/zhangzhong1/experiment/tianfugongyu/script/a_render_script/render/evaluate.py"

    echo "---- 渲染阶段 ----" | tee -a $LOGFILE
    blenderproc run "$RENDER_SCRIPT" --obj_folder "$OBJ_FOLDER" --xml_dir "$XML_DIR" >> $LOGFILE 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ 渲染成功：$name" | tee -a $LOGFILE
    else
        echo "❌ 渲染失败：$name" | tee -a $LOGFILE
    fi

    echo "---- 评估阶段 ----" | tee -a $LOGFILE
    python "$EVAL_SCRIPT" --input_folder "$OBJ_FOLDER" >> $LOGFILE 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ 评估成功：$name" | tee -a $LOGFILE
    else
        echo "❌ 评估失败：$name" | tee -a $LOGFILE
    fi

    echo -e "========== 完成数据集：$name ==========\n" | tee -a $LOGFILE
done

echo "========== 全部完成：$(date) ==========" | tee -a $LOGFILE